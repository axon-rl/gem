from pywikiapi import Site, AttrDict
from bs4 import BeautifulSoup

import re
import requests
from abc import ABC, abstractmethod
from requests.exceptions import Timeout
from typing import Dict, Optional
from warnings import warn

from .errors import DisambiguationException, QueryPageNotFoundException
from .wikipage import WikipediaPage

class BaseWikiTrawler(ABC):

    def __init__(
            self, 
            url: str = "https://en.wikipedia.org/w/api.php",
            query_delay_ms: int = 25,
            max_query_attempts: int = 5,
            query_use_cache: bool = False,

        ):
        self.url: str = url
        self.max_query_attempts: int = max_query_attempts
        self.query_use_cache: bool = query_use_cache
        self.query_delay_ms: int = query_delay_ms
        if self.query_use_cache:
            self._page_cache: Dict[str, WikipediaPage] = {}
    
    @abstractmethod
    def _direct_query(self, page_name: str) -> Optional[AttrDict]:
        pass

    @abstractmethod
    def _search_query(self, search_term: str) -> Optional[AttrDict]:
        pass

    @abstractmethod
    def random(self) -> Optional[WikipediaPage]:
        pass

    # Error handling behavior is adapted from ARENA 3.0, Chapter 3, Part 4.
    # Source: https://github.com/callummcdougall/ARENA_3.0/blob/main/chapter3_llm_evals/exercises/part4_llm_agents/3.4_LLM_Agents_solutions.ipynb
    def get_page(self, page_name: str) -> Optional[WikipediaPage]:
        if self.query_use_cache and page_name in self._page_cache:
            return self._page_cache[page_name]
        
        result: Optional[WikipediaPage] = None
        attempts = 0
        while result is None and attempts < self.max_query_attempts:
            try:
                result = self._direct_query(page_name)

                disambiguation = 'pageprops' in result['pages'][0] and 'disambiguation' in result['pages'][0]['pageprops']
                missing = 'missing' in result['pages'][0]
                if disambiguation:
                    raise DisambiguationException(
                        title = result['pages'][0]['title'],
                        options = [link['title'] for link in result['pages'][0]['links']]
                    )
                
                if missing:
                    raise QueryPageNotFoundException(result['pages'][0]['title'])

                page_object = result['pages'][0]
                result = WikipediaPage(
                    page_id = page_object['pageid'],
                    title = page_object['title'],
                    content = page_object['extract'],
                    links = [d['title'] for d in page_object['links']]
                )

            except DisambiguationException as e:
                # Prevent LLM cheesing the challenge by going to disambiguation pages
                # which don't quite reflect its world knowledge.
                # Subclass to remove this behavior.
                page_object = None
                # bugfix (171025): Sometimes we still get a red linked page.
                #                  Fixed by checking out all possible options.
                for option in e.options:
                    page_object = self._direct_query(option)['pages'][0]
                    if 'missing' not in page_object:
                        result = WikipediaPage(
                            page_id = page_object['pageid'],
                            title = page_object['title'],
                            content = page_object['extract'],
                            links = [d['title'] for d in page_object['links']]
                        )
                        break
                else:
                    result = None

            except QueryPageNotFoundException as e:
                # Use search as a fallback
                result = self._search_query(e.title)
                if not result['search']:
                    result = None
                else:
                    # Query again with "corrected" name.
                    result = self.get_page(result['search'][0]['title'])
            # APIError is deliberately unhandled because it should not occur
            except Exception as e:
                warn(f"Unexpected exception {e} occurred while querying for page '{page_name}'.")
                result = None
            attempts += 1

        # Don't cache None results; we want to retry them later.
        if self.query_use_cache and result is not None:
            self._page_cache[page_name] = result

        return result


class MediaWikiTrawler(BaseWikiTrawler):
    '''
    Uses live MediaWiki API calls to fetch pages,
    with help from the pywikiapi (https://github.com/nyurik/pywikiapi) package.
    '''

    def __init__(
            self, 
            url: str = "https://en.wikipedia.org/w/api.php",
            query_delay_ms: int = 25,
            max_query_attempts: int = 5,
            query_use_cache: bool = False,
        ):
        super().__init__(
            url, 
            query_delay_ms = query_delay_ms, 
            max_query_attempts = max_query_attempts, 
            query_use_cache = query_use_cache
        )

        # Define the site object
        self.site: Site = Site(
            self.url, 
            json_object_hook = AttrDict, 
            # pre_request_delay ingests seconds
            pre_request_delay = self.query_delay_ms / 1000.0
        )

    def _direct_query(self, page_name: str) -> Optional[AttrDict]:
        '''
        The base query function to fetch a page directly by its name.
        Allows redirects for simplicity's sake.
        '''
        return next(self.site.query(
            prop = 'extracts|links|info|pageprops',
            titles = page_name,
            explaintext = True,
            redirects = True,
            ppprop = 'disambiguation',
            plnamespace = 0,
        ))

    def _search_query(self, search_term: str) -> Optional[AttrDict]:
        '''
        A fallback query function to search for a page by a search term.
        '''
        return next(self.site.query(
            list = 'search',
            srsearch = search_term,
            srlimit = 1,
            srprop = '',
        ))

    def random(self) -> Optional[WikipediaPage]:
        '''
        Fetch a random Wikipedia page.
        '''
        result = next(self.site.query(
            list = 'random',
            rnnamespace = 0,
            rnlimit = 1,
        ))
        if not result['random']:
            return None
        return self.get_page(result['random'][0]['title'])
        
class KiwixWikiTrawler(BaseWikiTrawler):
    '''
    Fetches content from a local Kiwix endpoint.
    Unlike the MediaWiki endpoint, Kiwix is meant to be run locally with a pre-downloaded
    ZIM dump of Wikipedia.

    There are a few differences between Kiwix and MediaWiki wiki dumps:
    - Kiwix has fewer disambiguation pages
    - Kiwix's search engine is quite slow compared to MediaWiki's

    For implementation parity with MediaWikiTrawler, we try to mimic
    the same API structure as much as possible. This includes:
    - AttrDict return types
    - Analogous page resolution mechanisms
    and may come at the cost of performance / API complexity.

    WARNING: Inconsistencies have been anecdotally observed in Kiwix's 
        HTML article renders. It is possible for inputs to have a few
        extraneous and undesirable items such as:
        - Citation numbers in square brackets
        - "See also" sections
        - Various blurbs and notices
    '''

    def __init__(
            self,
            url: str = "http://localhost:8080", # Use HTTPS when feasible
            zimfile: str = "wikipedia_en_simple_all_nopic_2025-09", # WITHOUT .zim EXTENSION
            max_query_attempts: int = 5,
            query_delay_ms: int = 25,
            query_use_cache: bool = False,
    ):
        super().__init__(
            url, 
            max_query_attempts = max_query_attempts, 
            query_delay_ms = query_delay_ms, 
            query_use_cache = query_use_cache
        )
        self.zimfile: str = zimfile
        self.re_remove_space = re.compile(r'\s+')
        self.re_remove_newline = re.compile(r'\n+')

    def _direct_query(self, page_name: str) -> Optional[AttrDict]:
        try:
            response = requests.get(
                f"{self.url}/raw/{self.zimfile}/content/{page_name.replace(' ', '_')}",
                allow_redirects = True,
                timeout = 5
            )
        except Timeout as e:
            warn("Timeout occurred while querying Kiwix endpoint.")
            return AttrDict(
                pages = [AttrDict(
                    pageid = -1,
                    title = page_name,
                    missing = True,
                    extract = '',
                    links = []
                )]
            )

        if response.status_code == 404:
            raise QueryPageNotFoundException(page_name)
        raw_html = response.text

        # Use BS4 to extract the HTML content
        soup = BeautifulSoup(raw_html, 'html.parser')
        title = soup.select('h1.firstHeading')[0].text

        content_paras = soup.select('div.mw-content-ltr.mw-parser-output')[0]

        # Note that we retrieve the internal links before cleaning up the content paras.
        # This is because we want to reduce the prevalence of 'dead pages'
        # Delete cite notes and external links
        for tag in content_paras.select('a'):
            if (
                'href' not in tag.attrs
                or tag['href'].startswith('#')
                or tag['href'].startswith('http')
            ):
                tag.decompose()

        links = set(map(
            lambda lnk: lnk.get('href', None),
            content_paras.select('a')
        ))

        # Delete all textual content from first body para after references.
        ref_header = content_paras.find(id='References')
        if ref_header:
            for elem in ref_header.find_all_next():
                elem.decompose()
            ref_header.decompose()

        # Remove common unwanted tags
        for unwanted in [
            'table', '.infobox', '.navbox',
            '.vertical-navbox', '.thumb', '.thumbinner',
            'figure', '.mw-editsection', 'sup.reference',
            'style', 'script',
            '.sistersitebox', '.ambox', '.toc',
            '#toc', '.hatnote',
        ]:
            for tag in content_paras.select(unwanted):
                tag.decompose()

        content = "\n".join(para.get_text() for para in content_paras)
        # Finally, nuke excessive spaces and newlines
        content = self.re_remove_space.sub(' ', content)
        content = self.re_remove_newline.sub('\n', content)
        # Remove None links
        links.discard(None)

        query_result = AttrDict(
            pages = [AttrDict(
                pageid = title,
                title = title,
                extract = content,
                links = [AttrDict(title = lnk) for lnk in links]
            )]
        )
        
        if 'may refer to:' in content.lower() or 'disambiguation' in title.lower():
            query_result['pages'][0]['pageprops'] = AttrDict(disambiguation = True)
        
        return query_result

    def _search_query(self, search_term: str) -> Optional[AttrDict]:
        '''
        For convenience, we let the search endpoint only return
        the first result.
        '''
        try:
            result = requests.get(
                f"{self.url}/search?pattern={search_term}",
                timeout = 5
            )
        except Timeout as e:
            warn("Timeout occurred while querying Kiwix endpoint.")
            return AttrDict(
                search = []
            )
        
        raw_html = result.text
        soup = BeautifulSoup(raw_html, 'html.parser')
        search_results = soup.select('div.results.ul.li.a')
        return AttrDict(
            search = [AttrDict(
                title = search_results[0].text.strip()
            )] if search_results else []
        )

    def random(self) -> Optional[WikipediaPage]:
        response: Optional[requests.Response] = None
        for _ in range(self.max_query_attempts):
            try:
                response = requests.get(
                    f"{self.url}/random?content={self.zimfile}",
                    allow_redirects = True,
                    timeout = 5
                )
                print(f'{self.url}/random?content={self.zimfile}: {response.status_code}')
            except Timeout as e:
                warn("Timeout occurred while querying Kiwix endpoint.")
                continue
            except Exception as e:
                print(f"Unexpected exception {e} occurred while querying for a random page.")
                continue
            if response.status_code == 200:
                break
        else:
            return None
        return self.get_page(response.url.split('/')[-1])