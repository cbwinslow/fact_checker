from __future__ import annotations

import asyncio
import logging
import re
import urllib.parse
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

import httpx

from ..models import EvidenceItem
from ..config import get_settings

# Optional BS4 import
try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False
    BeautifulSoup = None  # type: ignore

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


class SearchResult:
    """Normalized search result from any provider."""
    
    def __init__(
        self,
        url: str,
        title: str,
        snippet: str,
        domain: str,
        published_date: Optional[datetime] = None,
        score: float = 0.5,
        provider: str = "",
        raw_data: Optional[Dict] = None,
        source_type: str = "other",
        author: Optional[str] = None,
    ):
        self.url = url
        self.title = title
        self.snippet = snippet
        self.domain = domain
        self.published_date = published_date
        self.score = score
        self.provider = provider
        self.raw_data = raw_data or {}
        self.source_type = source_type
        self.author = author
    
    def to_evidence_item(self, claim_id: UUID) -> EvidenceItem:
        """Convert to EvidenceItem for pipeline integration."""
        return EvidenceItem(
            claim_id=claim_id,
            source_url=self.url,
            title=self.title,
            snippet=self.snippet,
            relevance_score=self.score,
            domain=self.domain,
            published_date=self.published_date,
            author=self.author,
            source_type=self._infer_source_type(),
        )
    
    def _infer_source_type(self) -> str:
        domain = self.domain.lower()
        if any(d in domain for d in ["snopes.com", "politifact.com", "factcheck.org", "reuters.com/fact-check"]):
            return "factcheck"
        if any(d in domain for d in ["arxiv.org", "semantic-scholar.org", "pubmed.ncbi.nlm.nih.gov", "doi.org"]):
            return "academic"
        if any(d in domain for d in [".gov", ".mil", "congress.gov", "govinfo.gov"]):
            return "government"
        if "wikipedia.org" in domain or "wikidata.org" in domain:
            return "wiki"
        if any(d in domain for d in ["nytimes.com", "washingtonpost.com", "theguardian.com", "bbc.com", "reuters.com", "apnews.com"]):
            return "news"
        return "other"


class Quote:
    """An exact quote extracted from source text with context."""
    
    def __init__(
        self,
        text: str,
        context_before: str = "",
        context_after: str = "",
        offset: int = 0,
        relevance_score: float = 0.0,
    ):
        self.text = text
        self.context_before = context_before
        self.context_after = context_after
        self.offset = offset
        self.relevance_score = relevance_score
    
    @property
    def full_context(self) -> str:
        return f"{self.context_before}{self.text}{self.context_after}"


# ---------------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------------


class SearchProvider(ABC):
    """Abstract base class for search providers."""
    
    name: str = "base"
    source_type: str = "general"
    rate_limit_rps: float = 1.0  # requests per second
    
    def __init__(self):
        self._last_request = 0.0
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers={"User-Agent": "FactCheckerBot/1.0 (+https://github.com/cbwinslow/fact_checker)"},
            )
        return self._client
    
    async def _rate_limit(self):
        """Enforce rate limiting."""
        import time
        now = time.monotonic()
        min_interval = 1.0 / self.rate_limit_rps
        elapsed = now - self._last_request
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        self._last_request = time.monotonic()
    
    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search for query and return normalized results."""
        pass
    
    @abstractmethod
    async def fetch_full(self, url: str) -> Optional[str]:
        """Fetch full content from a URL."""
        pass
    
    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# ---------------------------------------------------------------------------
# Provider Implementations
# ---------------------------------------------------------------------------


class DuckDuckGoProvider(SearchProvider):
    """DuckDuckGo HTML search (no API key required)."""
    
    name = "duckduckgo"
    source_type = "general"
    rate_limit_rps = 0.5  # Be respectful
    
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        await self._rate_limit()
        client = await self._get_client()
        
        url = "https://html.duckduckgo.com/html/"
        params = {"q": query, "kl": "us-en"}
        
        try:
            resp = await client.post(url, data=params)
            resp.raise_for_status()
        except Exception as exc:
            log.warning(f"[{self.name}] Search failed: {exc}")
            return []
        
        soup = BeautifulSoup(resp.text, "html.parser")
        results: List[SearchResult] = []
        
        for result in soup.select(".result__body")[:max_results]:
            try:
                link = result.select_one(".result__url")
                title_elem = result.select_one(".result__title a")
                snippet_elem = result.select_one(".result__snippet")
                
                if not title_elem:
                    continue
                
                url = title_elem.get("href", "")
                title = title_elem.get_text(strip=True)
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                domain = link.get_text(strip=True) if link else urllib.parse.urlparse(url).netloc
                
                results.append(SearchResult(
                    url=url,
                    title=title,
                    snippet=snippet,
                    domain=domain,
                    score=0.6,
                    provider=self.name,
                ))
            except Exception as exc:
                log.debug(f"[{self.name}] Failed to parse result: {exc}")
                continue
        
        log.info(f"[{self.name}] Found {len(results)} results for: {query[:50]}")
        return results
    
    async def fetch_full(self, url: str) -> Optional[str]:
        await self._rate_limit()
        client = await self._get_client()
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            log.warning(f"[{self.name}] Fetch failed for {url}: {exc}")
            return None


class BraveSearchProvider(SearchProvider):
    """Brave Search API (free tier: 2000 requests/month)."""
    
    name = "brave"
    source_type = "general"
    rate_limit_rps = 1.0
    
    def __init__(self):
        super().__init__()
        self.api_key = getattr(settings, "brave_search_api_key", "").strip()
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
    
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        if not self.api_key:
            log.debug(f"[{self.name}] No API key configured, skipping")
            return []
        
        await self._rate_limit()
        client = await self._get_client()
        
        params = {
            "q": query,
            "count": min(max_results, 20),
            "search_lang": "en",
            "country": "US",
            "safesearch": "moderate",
        }
        headers = {"X-Subscription-Token": self.api_key, "Accept": "application/json"}
        
        try:
            resp = await client.get(self.base_url, params=params, headers=headers)
            if resp.status_code == 429:
                log.warning(f"[{self.name}] Rate limited")
                return []
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning(f"[{self.name}] Search failed: {exc}")
            return []
        
        results: List[SearchResult] = []
        for item in data.get("web", {}).get("results", [])[:max_results]:
            try:
                url = item.get("url", "")
                domain = urllib.parse.urlparse(url).netloc
                published = None
                if item.get("age"):
                    # Parse age string like "2 hours ago"
                    published = self._parse_age(item["age"])
                
                results.append(SearchResult(
                    url=url,
                    title=item.get("title", ""),
                    snippet=item.get("description", ""),
                    domain=domain,
                    published_date=published,
                    score=0.65,
                    provider=self.name,
                    source_type="general",
                    raw_data=item,
                ))
            except Exception as exc:
                log.debug(f"[{self.name}] Failed to parse result: {exc}")
                continue
        
        log.info(f"[{self.name}] Found {len(results)} results for: {query[:50]}")
        return results
    
    def _parse_age(self, age_str: str) -> Optional[datetime]:
        """Parse age strings like '2 hours ago', '3 days ago'."""
        # Simplified - in production use dateparser library
        return None
    
    async def fetch_full(self, url: str) -> Optional[str]:
        return await DuckDuckGoProvider().fetch_full(url)  # Reuse generic fetcher


class SemanticScholarProvider(SearchProvider):
    """Semantic Scholar API (free, generous limits)."""
    
    name = "semantic_scholar"
    source_type = "academic"
    rate_limit_rps = 10.0  # 100 req/5min
    
    def __init__(self):
        super().__init__()
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.api_key = getattr(settings, "semantic_scholar_api_key", "").strip()  # Optional
    
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        await self._rate_limit()
        client = await self._get_client()
        
        url = f"{self.base_url}/paper/search"
        params = {
            "query": query,
            "limit": min(max_results, 100),
            "fields": "title,url,venue,year,authors,abstract,citationCount,openAccessPdf",
        }
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        
        try:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning(f"[{self.name}] Search failed: {exc}")
            return []
        
        results: List[SearchResult] = []
        for paper in data.get("data", [])[:max_results]:
            try:
                paper_id = paper.get("paperId", "")
                url = paper.get("url", f"https://www.semanticscholar.org/paper/{paper_id}")
                authors = ", ".join(a.get("name", "") for a in paper.get("authors", [])[:3])
                year = paper.get("year")
                published = datetime(year, 1, 1) if year else None
                
                results.append(SearchResult(
                    url=url,
                    title=paper.get("title", ""),
                    snippet=paper.get("abstract", "")[:300],
                    domain="semanticscholar.org",
                    published_date=published,
                    author=authors,
                    score=0.7,
                    provider=self.name,
                    source_type="academic",
                    raw_data=paper,
                ))
            except Exception as exc:
                log.debug(f"[{self.name}] Failed to parse result: {exc}")
                continue
        
        log.info(f"[{self.name}] Found {len(results)} results for: {query[:50]}")
        return results
    
    async def fetch_full(self, url: str) -> Optional[str]:
        # For Semantic Scholar, we already have abstract in search results
        # Could fetch full PDF if openAccessPdf available
        return None


class ArxivProvider(SearchProvider):
    """Arxiv API (free, no key required)."""
    
    name = "arxiv"
    source_type = "academic"
    rate_limit_rps = 0.33  # 1 request per 3 seconds
    
    def __init__(self):
        super().__init__()
        self.base_url = "http://export.arxiv.org/api/query"
    
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        await self._rate_limit()
        client = await self._get_client()
        
        params = {
            "search_query": query,
            "start": 0,
            "max_results": min(max_results, 50),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        
        try:
            resp = await client.get(self.base_url, params=params)
            resp.raise_for_status()
            xml_text = resp.text
        except Exception as exc:
            log.warning(f"[{self.name}] Search failed: {exc}")
            return []
        
        soup = BeautifulSoup(xml_text, "xml")
        results: List[SearchResult] = []
        
        for entry in soup.find_all("entry")[:max_results]:
            try:
                arxiv_id = entry.find("id").text.split("/")[-1] if entry.find("id") else ""
                title = entry.find("title").text.strip() if entry.find("title") else ""
                summary = entry.find("summary").text.strip() if entry.find("summary") else ""
                published_str = entry.find("published").text if entry.find("published") else ""
                authors = [a.find("name").text for a in entry.find_all("author") if a.find("name")]
                url = f"https://arxiv.org/abs/{arxiv_id}"
                
                published = None
                if published_str:
                    try:
                        published = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                    except Exception:
                        pass
                
                results.append(SearchResult(
                    url=url,
                    title=title,
                    snippet=summary[:300],
                    domain="arxiv.org",
                    published_date=published,
                    author=", ".join(authors[:3]),
                    score=0.75,
                    provider=self.name,
                    source_type="academic",
                ))
            except Exception as exc:
                log.debug(f"[{self.name}] Failed to parse result: {exc}")
                continue
        
        log.info(f"[{self.name}] Found {len(results)} results for: {query[:50]}")
        return results
    
    async def fetch_full(self, url: str) -> Optional[str]:
        # Could fetch PDF, but for now return None (abstract in search)
        return None


class PubMedProvider(SearchProvider):
    """PubMed/NCBI E-utilities (free, requires email for high volume)."""
    
    name = "pubmed"
    source_type = "academic"
    rate_limit_rps = 3.0  # 3 req/sec without key
    
    def __init__(self):
        super().__init__()
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.email = getattr(settings, "pubmed_email", "factchecker@example.com")
    
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        await self._rate_limit()
        client = await self._get_client()
        
        # Step 1: Search for PMIDs
        search_url = f"{self.base_url}/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": min(max_results, 100),
            "retmode": "json",
            "email": self.email,
        }
        
        try:
            resp = await client.get(search_url, params=search_params)
            resp.raise_for_status()
            search_data = resp.json()
        except Exception as exc:
            log.warning(f"[{self.name}] Search failed: {exc}")
            return []
        
        pmids = search_data.get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []
        
        # Step 2: Fetch details for PMIDs
        await self._rate_limit()
        fetch_url = f"{self.base_url}/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids[:max_results]),
            "retmode": "xml",
            "email": self.email,
        }
        
        try:
            resp = await client.get(fetch_url, params=fetch_params)
            resp.raise_for_status()
            xml_text = resp.text
        except Exception as exc:
            log.warning(f"[{self.name}] Fetch failed: {exc}")
            return []
        
        soup = BeautifulSoup(xml_text, "xml")
        results: List[SearchResult] = []
        
        for article in soup.find_all("PubmedArticle")[:max_results]:
            try:
                pmid = article.find("PMID").text if article.find("PMID") else ""
                title_elem = article.find("ArticleTitle")
                title = title_elem.text.strip() if title_elem else ""
                
                abstract_elem = article.find("AbstractText")
                abstract = abstract_elem.text.strip() if abstract_elem else ""
                
                # Authors
                authors = []
                for author in article.find_all("Author"):
                    last = author.find("LastName")
                    first = author.find("ForeName")
                    if last and first:
                        authors.append(f"{first.text} {last.text}")
                    elif last:
                        authors.append(last.text)
                
                # Publication date
                pub_date = article.find("PubDate")
                published = None
                if pub_date:
                    year = pub_date.find("Year")
                    month = pub_date.find("Month")
                    day = pub_date.find("Day")
                    if year:
                        try:
                            m = int(month.text) if month and month.text.isdigit() else 1
                            d = int(day.text) if day and day.text.isdigit() else 1
                            published = datetime(int(year.text), m, d)
                        except Exception:
                            pass
                
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                
                results.append(SearchResult(
                    url=url,
                    title=title,
                    snippet=abstract[:300],
                    domain="pubmed.ncbi.nlm.nih.gov",
                    published_date=published,
                    author=", ".join(authors[:3]),
                    score=0.8,
                    provider=self.name,
                    source_type="academic",
                ))
            except Exception as exc:
                log.debug(f"[{self.name}] Failed to parse result: {exc}")
                continue
        
        log.info(f"[{self.name}] Found {len(results)} results for: {query[:50]}")
        return results
    
    async def fetch_full(self, url: str) -> Optional[str]:
        # Could fetch full XML for a PMID
        return None


class CrossrefProvider(SearchProvider):
    """Crossref REST API (free, DOI metadata)."""
    
    name = "crossref"
    source_type = "academic"
    rate_limit_rps = 5.0  # Polite limit
    
    def __init__(self):
        super().__init__()
        self.base_url = "https://api.crossref.org/works"
    
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        await self._rate_limit()
        client = await self._get_client()
        
        params = {
            "query": query,
            "rows": min(max_results, 20),
            "select": "DOI,title,author,container-title,published-online,abstract,URL,type",
        }
        headers = {"User-Agent": "FactCheckerBot/1.0 (mailto:factchecker@example.com)"}
        
        try:
            resp = await client.get(self.base_url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning(f"[{self.name}] Search failed: {exc}")
            return []
        
        results: List[SearchResult] = []
        for item in data.get("message", {}).get("items", [])[:max_results]:
            try:
                doi = item.get("DOI", "")
                url = item.get("URL", f"https://doi.org/{doi}")
                title = item.get("title", [""])[0] if item.get("title") else ""
                container = item.get("container-title", [""])[0] if item.get("container-title") else ""
                abstract = item.get("abstract", "")
                
                authors = []
                for author in item.get("author", [])[:3]:
                    given = author.get("given", "")
                    family = author.get("family", "")
                    if given or family:
                        authors.append(f"{given} {family}".strip())
                
                published = None
                pub_online = item.get("published-online", {})
                if pub_online and "date-parts" in pub_online:
                    parts = pub_online["date-parts"][0]
                    if parts:
                        try:
                            y, m, d = parts[0], parts[1] if len(parts) > 1 else 1, parts[2] if len(parts) > 2 else 1
                            published = datetime(y, m, d)
                        except Exception:
                            pass
                
                snippet = abstract[:300] if abstract else ""
                if container:
                    snippet = f"[{container}] {snippet}"
                
                results.append(SearchResult(
                    url=url,
                    title=title,
                    snippet=snippet,
                    domain="doi.org",
                    published_date=published,
                    author=", ".join(authors),
                    score=0.75,
                    provider=self.name,
                    source_type="academic",
                ))
            except Exception as exc:
                log.debug(f"[{self.name}] Failed to parse result: {exc}")
                continue
        
        log.info(f"[{self.name}] Found {len(results)} results for: {query[:50]}")
        return results
    
    async def fetch_full(self, url: str) -> Optional[str]:
        return None


class GovInfoProvider(SearchProvider):
    """GovInfo API (US Government documents, free)."""
    
    name = "govinfo"
    source_type = "government"
    rate_limit_rps = 2.0
    
    def __init__(self):
        super().__init__()
        self.base_url = "https://api.govinfo.gov/search"
        self.api_key = getattr(settings, "govinfo_api_key", "").strip()  # Optional but recommended
    
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        await self._rate_limit()
        client = await self._get_client()
        
        params = {
            "query": query,
            "pageSize": min(max_results, 100),
            "offsetMark": "*",
        }
        if self.api_key:
            params["api_key"] = self.api_key
        
        try:
            resp = await client.get(self.base_url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning(f"[{self.name}] Search failed: {exc}")
            return []
        
        results: List[SearchResult] = []
        for item in data.get("results", [])[:max_results]:
            try:
                url = item.get("downloadUrl", item.get("packageLink", ""))
                title = item.get("title", "")
                snippet = item.get("summary", "")[:300]
                published_str = item.get("dateIssued", "")
                
                published = None
                if published_str:
                    try:
                        published = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                    except Exception:
                        pass
                
                results.append(SearchResult(
                    url=url,
                    title=title,
                    snippet=snippet,
                    domain="govinfo.gov",
                    published_date=published,
                    score=0.85,
                    provider=self.name,
                    source_type="government",
                ))
            except Exception as exc:
                log.debug(f"[{self.name}] Failed to parse result: {exc}")
                continue
        
        log.info(f"[{self.name}] Found {len(results)} results for: {query[:50]}")
        return results
    
    async def fetch_full(self, url: str) -> Optional[str]:
        return None


class WikipediaProvider(SearchProvider):
    """Wikipedia REST API (already used in deep_research_agent)."""
    
    name = "wikipedia"
    source_type = "wiki"
    rate_limit_rps = 10.0
    
    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        await self._rate_limit()
        client = await self._get_client()
        
        # Use opensearch for search
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "opensearch",
            "search": query,
            "limit": max_results,
            "format": "json",
        }
        
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning(f"[{self.name}] Search failed: {exc}")
            return []
        
        results: List[SearchResult] = []
        titles = data[1] if len(data) > 1 else []
        descriptions = data[2] if len(data) > 2 else []
        urls = data[3] if len(data) > 3 else []
        
        for title, desc, url in zip(titles, descriptions, urls):
            results.append(SearchResult(
                url=url,
                title=title,
                snippet=desc[:300],
                domain="wikipedia.org",
                score=0.7,
                provider=self.name,
                source_type="wiki",
            ))
        
        return results
    
    async def fetch_full(self, url: str) -> Optional[str]:
        """Fetch full page content via REST API."""
        # Extract title from URL
        title = url.split("/")[-1].replace("_", " ")
        await self._rate_limit()
        client = await self._get_client()
        
        api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
        try:
            resp = await client.get(api_url)
            resp.raise_for_status()
            data = resp.json()
            return data.get("extract", "")
        except Exception as exc:
            log.warning(f"[{self.name}] Fetch failed: {exc}")
            return None


# ---------------------------------------------------------------------------
# Provider Registry
# ---------------------------------------------------------------------------


class ProviderRegistry:
    """Registry of available search providers."""
    
    def __init__(self):
        self._providers: Dict[str, SearchProvider] = {}
        self._register_defaults()
    
    def _register_defaults(self):
        # Register all free providers
        self.register(DuckDuckGoProvider())
        self.register(BraveSearchProvider())
        self.register(SemanticScholarProvider())
        self.register(ArxivProvider())
        self.register(PubMedProvider())
        self.register(CrossrefProvider())
        self.register(GovInfoProvider())
        self.register(WikipediaProvider())
    
    def register(self, provider: SearchProvider):
        self._providers[provider.name] = provider
        log.info(f"[registry] Registered provider: {provider.name} ({provider.source_type})")
    
    def get(self, name: str) -> Optional[SearchProvider]:
        return self._providers.get(name)
    
    def get_by_type(self, source_type: str) -> List[SearchProvider]:
        return [p for p in self._providers.values() if p.source_type == source_type]
    
    def all(self) -> List[SearchProvider]:
        return list(self._providers.values())
    
    async def search_all(
        self,
        query: str,
        max_results_per_provider: int = 5,
        source_types: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Search across all providers (or filtered by type)."""
        providers = list(self._providers.values())
        if source_types:
            providers = [p for p in providers if p.source_type in source_types]
        
        tasks = [p.search(query, max_results_per_provider) for p in providers]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        combined: List[SearchResult] = []
        for provider, results in zip(providers, all_results):
            if isinstance(results, Exception):
                log.warning(f"[{provider.name}] Search error: {results}")
                continue
            combined.extend(results)
        
        # Sort by score descending
        combined.sort(key=lambda r: r.score, reverse=True)
        return combined
    
    async def close_all(self):
        for provider in self._providers.values():
            await provider.close()


# Global registry instance
_registry: Optional[ProviderRegistry] = None


def get_registry() -> ProviderRegistry:
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry


# ---------------------------------------------------------------------------
# Quote Extraction
# ---------------------------------------------------------------------------


def extract_quotes(
    text: str,
    claim_text: str,
    max_quotes: int = 3,
    context_chars: int = 200,
) -> List[Quote]:
    """Extract claim-relevant quotes from source text.
    
    Strategy:
    1. Split text into sentences
    2. Score each sentence for relevance to claim (entity/keyword overlap)
    3. Return top-k with surrounding context
    """
    if not text or not claim_text:
        return []
    
    # Normalize claim for matching
    claim_terms = set(re.findall(r"\b[a-z]{3,}\b", claim_text.lower()))
    claim_entities = set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", claim_text))
    
    # Split into sentences (simple heuristic)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    
    scored: List[tuple[float, str, int]] = []  # (score, sentence, offset)
    current_offset = 0
    
    for sent in sentences:
        sent_lower = sent.lower()
        sent_terms = set(re.findall(r"\b[a-z]{3,}\b", sent_lower))
        sent_entities = set(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", sent))
        
        # Score: term overlap + entity match
        term_overlap = len(claim_terms & sent_terms) / max(len(claim_terms), 1)
        entity_match = len(claim_entities & sent_entities) / max(len(claim_entities), 1) if claim_entities else 0
        
        score = term_overlap * 0.7 + entity_match * 0.3
        
        if score > 0:
            scored.append((score, sent.strip(), current_offset))
        
        current_offset += len(sent) + 1
    
    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    
    quotes: List[Quote] = []
    for score, sent, offset in scored[:max_quotes]:
        # Get context
        start = max(0, offset - context_chars)
        end = min(len(text), offset + len(sent) + context_chars)
        context_before = text[start:offset]
        context_after = text[offset + len(sent):end]
        
        quotes.append(Quote(
            text=sent,
            context_before=context_before,
            context_after=context_after,
            offset=offset,
            relevance_score=score,
        ))
    
    return quotes


async def enrich_search_results_with_quotes(
    results: List[SearchResult],
    claim_text: str,
    registry: ProviderRegistry,
) -> List[SearchResult]:
    """Fetch full content for results and extract quotes."""
    enriched = []
    
    for result in results:
        provider = registry.get(result.provider)
        if not provider:
            enriched.append(result)
            continue
        
        full_text = await provider.fetch_full(result.url)
        if full_text:
            quotes = extract_quotes(full_text, claim_text)
            if quotes:
                best_quote = max(quotes, key=lambda q: q.relevance_score)
                result.raw_data["quotes"] = [
                    {
                        "text": q.text,
                        "context_before": q.context_before,
                        "context_after": q.context_after,
                        "offset": q.offset,
                        "relevance_score": q.relevance_score,
                    }
                    for q in quotes
                ]
                result.raw_data["full_text"] = full_text[:5000]  # Cap for storage
        
        enriched.append(result)
    
    return enriched