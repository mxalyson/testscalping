"""
News Sentiment Filter - Filtro de Notícias e Sentiment Analysis
===============================================================

Módulo para:
- Monitorar notícias de criptomoedas em tempo real
- Analisar sentimento (positivo/negativo/neutro)
- Filtrar trades em eventos de alto impacto
- Integração com APIs de notícias
- Cache e rate limiting

APIs suportadas:
- CryptoPanic (gratuita com limites)
- NewsAPI (requer key)
- Twitter/X API (opcional)
- RSS Feeds de cripto

Autor: Trading Bot Team
"""

import requests
import time
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json
from pathlib import Path
from loguru import logger
import re
from enum import Enum

# Sentiment analysis (opcional - instalar se disponível)
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False
    logger.warning("TextBlob não instalado. Use: pip install textblob")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False
    logger.warning("VADER não instalado. Use: pip install vaderSentiment")


# ============================================================================
# ENUMS E CONSTANTES
# ============================================================================

class SentimentLabel(Enum):
    """Labels de sentimento"""
    VERY_POSITIVE = "VERY_POSITIVE"
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"
    VERY_NEGATIVE = "VERY_NEGATIVE"


class NewsImpact(Enum):
    """Nível de impacto da notícia"""
    CRITICAL = "CRITICAL"  # Halt trading
    HIGH = "HIGH"          # Aguardar estabilização
    MEDIUM = "MEDIUM"      # Considerar no contexto
    LOW = "LOW"            # Ignorar


# Keywords para detectar eventos importantes
CRITICAL_KEYWORDS = [
    "hack", "hacked", "exploit", "vulnerability", "security breach",
    "sec charges", "lawsuit", "banned", "regulation",
    "exchange down", "outage", "crash", "delisted"
]

HIGH_IMPACT_KEYWORDS = [
    "sec", "regulation", "approval", "etf", "institutional",
    "adoption", "partnership", "upgrade", "fork",
    "halving", "listing", "exchange"
]

NEGATIVE_KEYWORDS = [
    "crash", "dump", "sell-off", "bearish", "decline", "drop",
    "fear", "panic", "scam", "fraud", "ponzi"
]

POSITIVE_KEYWORDS = [
    "rally", "surge", "bullish", "gain", "pump", "moon",
    "breakthrough", "adoption", "innovation"
]


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class NewsArticle:
    """Representa uma notícia"""
    id: str
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    symbols: List[str] = field(default_factory=list)
    sentiment: Optional[SentimentLabel] = None
    sentiment_score: float = 0.0
    impact_level: Optional[NewsImpact] = None
    keywords_found: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content[:200] + '...' if len(self.content) > 200 else self.content,
            'source': self.source,
            'url': self.url,
            'published_at': self.published_at.isoformat(),
            'symbols': self.symbols,
            'sentiment': self.sentiment.value if self.sentiment else None,
            'sentiment_score': self.sentiment_score,
            'impact_level': self.impact_level.value if self.impact_level else None,
            'keywords_found': self.keywords_found
        }


@dataclass
class NewsFilter:
    """Resultado do filtro de notícias"""
    should_trade: bool
    reason: str
    recent_news: List[NewsArticle]
    overall_sentiment: SentimentLabel
    sentiment_score: float
    critical_events: List[NewsArticle]


# ============================================================================
# CLASSE PRINCIPAL: NewsSentimentFilter
# ============================================================================

class NewsSentimentFilter:
    """
    Filtro de notícias e análise de sentimento
    """
    
    def __init__(
        self,
        cryptopanic_api_key: Optional[str] = None,
        newsapi_key: Optional[str] = None,
        cache_hours: int = 1,
        max_cache_size: int = 500,
        enable_sentiment_analysis: bool = True,
        sentiment_engine: Literal["textblob", "vader", "keywords"] = "vader"
    ):
        """
        Inicializa o filtro de notícias
        
        Args:
            cryptopanic_api_key: API key do CryptoPanic
            newsapi_key: API key do NewsAPI
            cache_hours: Horas para manter notícias em cache
            max_cache_size: Tamanho máximo do cache
            enable_sentiment_analysis: Se True, analisa sentimento
            sentiment_engine: Engine de análise de sentimento
        """
        self.cryptopanic_api_key = cryptopanic_api_key
        self.newsapi_key = newsapi_key
        self.cache_hours = cache_hours
        self.max_cache_size = max_cache_size
        self.enable_sentiment_analysis = enable_sentiment_analysis
        self.sentiment_engine = sentiment_engine
        
        # Cache de notícias
        self.news_cache: deque = deque(maxlen=max_cache_size)
        self.last_fetch_time: Dict[str, datetime] = {}
        
        # Rate limiting
        self.rate_limit_delay = 2  # segundos entre requests
        self.last_request_time = 0
        
        # Sentiment analyzer
        self.vader_analyzer = None
        if sentiment_engine == "vader" and HAS_VADER:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Estatísticas
        self.total_news_fetched = 0
        self.critical_events_detected = 0
        
        logger.info(
            f"NewsSentimentFilter inicializado: "
            f"engine={sentiment_engine}, cache={cache_hours}h"
        )
    
    # ========================================================================
    # FETCH DE NOTÍCIAS
    # ========================================================================
    
    def fetch_news(
        self,
        symbols: List[str],
        hours_back: int = 6,
        force_refresh: bool = False
    ) -> List[NewsArticle]:
        """
        Busca notícias para os símbolos especificados
        
        Args:
            symbols: Lista de símbolos (ex: ["BTC", "ETH"])
            hours_back: Horas passadas a buscar
            force_refresh: Se True, ignora cache
        
        Returns:
            Lista de NewsArticle
        """
        # Normaliza símbolos
        symbols = [s.replace("USDT", "").upper() for s in symbols]
        
        # Verifica cache
        if not force_refresh:
            cached_news = self._get_cached_news(symbols, hours_back)
            if cached_news:
                logger.debug(f"Retornando {len(cached_news)} notícias do cache")
                return cached_news
        
        # Busca de múltiplas fontes
        all_news = []
        
        # 1. CryptoPanic
        if self.cryptopanic_api_key:
            try:
                news = self._fetch_cryptopanic(symbols)
                all_news.extend(news)
            except Exception as e:
                logger.error(f"Erro ao buscar CryptoPanic: {e}")
        
        # 2. NewsAPI
        if self.newsapi_key:
            try:
                news = self._fetch_newsapi(symbols, hours_back)
                all_news.extend(news)
            except Exception as e:
                logger.error(f"Erro ao buscar NewsAPI: {e}")
        
        # 3. RSS Feeds (gratuito)
        try:
            news = self._fetch_rss_feeds(symbols, hours_back)
            all_news.extend(news)
        except Exception as e:
            logger.error(f"Erro ao buscar RSS: {e}")
        
        # Remove duplicatas (por URL)
        unique_news = self._deduplicate_news(all_news)
        
        # Analisa sentimento
        if self.enable_sentiment_analysis:
            for article in unique_news:
                self._analyze_sentiment(article)
                self._classify_impact(article)
        
        # Atualiza cache
        self.news_cache.extend(unique_news)
        self.total_news_fetched += len(unique_news)
        
        logger.info(f"📰 Buscadas {len(unique_news)} notícias para {symbols}")
        
        return unique_news
    
    def _fetch_cryptopanic(self, symbols: List[str]) -> List[NewsArticle]:
        """Busca notícias do CryptoPanic"""
        self._rate_limit()
        
        url = "https://cryptopanic.com/api/v1/posts/"
        params = {
            "auth_token": self.cryptopanic_api_key,
            "currencies": ",".join(symbols),
            "kind": "news",
            "filter": "hot"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        articles = []
        
        for item in data.get("results", []):
            article = NewsArticle(
                id=f"cp_{item['id']}",
                title=item['title'],
                content=item.get('body', item['title']),
                source=item.get('source', {}).get('title', 'CryptoPanic'),
                url=item['url'],
                published_at=datetime.fromisoformat(item['published_at'].replace('Z', '+00:00')),
                symbols=[c['code'] for c in item.get('currencies', [])]
            )
            articles.append(article)
        
        return articles
    
    def _fetch_newsapi(self, symbols: List[str], hours_back: int) -> List[NewsArticle]:
        """Busca notícias do NewsAPI"""
        self._rate_limit()
        
        # Constrói query
        queries = [f"{symbol} OR cryptocurrency" for symbol in symbols]
        query = " OR ".join(queries)
        
        from_date = (datetime.now() - timedelta(hours=hours_back)).isoformat()
        
        url = "https://newsapi.org/v2/everything"
        params = {
            "apiKey": self.newsapi_key,
            "q": query,
            "from": from_date,
            "language": "en",
            "sortBy": "publishedAt"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        articles = []
        
        for item in data.get("articles", []):
            article = NewsArticle(
                id=f"na_{hash(item['url'])}",
                title=item['title'],
                content=item.get('description', '') + ' ' + item.get('content', ''),
                source=item['source']['name'],
                url=item['url'],
                published_at=datetime.fromisoformat(item['publishedAt'].replace('Z', '+00:00')),
                symbols=self._extract_symbols(item['title'] + ' ' + item.get('description', ''))
            )
            articles.append(article)
        
        return articles
    
    def _fetch_rss_feeds(self, symbols: List[str], hours_back: int) -> List[NewsArticle]:
        """Busca notícias de RSS feeds (gratuito)"""
        # Feeds públicos de cripto
        feeds = [
            "https://cointelegraph.com/rss",
            "https://cryptonews.com/news/feed/",
            "https://bitcoinmagazine.com/feed"
        ]
        
        articles = []
        
        try:
            import feedparser
        except ImportError:
            logger.warning("feedparser não instalado. Use: pip install feedparser")
            return articles
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        for feed_url in feeds:
            try:
                self._rate_limit()
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:20]:  # Limita a 20 por feed
                    # Parse data
                    published = entry.get('published_parsed')
                    if published:
                        pub_date = datetime(*published[:6])
                        if pub_date < cutoff_time:
                            continue
                    else:
                        pub_date = datetime.now()
                    
                    # Verifica se menciona símbolos
                    text = entry.title + ' ' + entry.get('summary', '')
                    mentioned_symbols = self._extract_symbols(text)
                    
                    if not any(s in mentioned_symbols for s in symbols):
                        continue
                    
                    article = NewsArticle(
                        id=f"rss_{hash(entry.link)}",
                        title=entry.title,
                        content=entry.get('summary', ''),
                        source=feed.feed.get('title', 'RSS'),
                        url=entry.link,
                        published_at=pub_date,
                        symbols=mentioned_symbols
                    )
                    articles.append(article)
            
            except Exception as e:
                logger.error(f"Erro ao processar feed {feed_url}: {e}")
                continue
        
        return articles
    
    # ========================================================================
    # ANÁLISE DE SENTIMENTO
    # ========================================================================
    
    def _analyze_sentiment(self, article: NewsArticle):
        """Analisa sentimento de uma notícia"""
        text = article.title + ' ' + article.content
        
        if self.sentiment_engine == "vader" and self.vader_analyzer:
            score = self._sentiment_vader(text)
        elif self.sentiment_engine == "textblob" and HAS_TEXTBLOB:
            score = self._sentiment_textblob(text)
        else:
            score = self._sentiment_keywords(text)
        
        article.sentiment_score = score
        article.sentiment = self._score_to_label(score)
    
    def _sentiment_vader(self, text: str) -> float:
        """Sentimento usando VADER"""
        scores = self.vader_analyzer.polarity_scores(text)
        # Composto vai de -1 a 1
        return scores['compound']
    
    def _sentiment_textblob(self, text: str) -> float:
        """Sentimento usando TextBlob"""
        blob = TextBlob(text)
        # Polarity vai de -1 a 1
        return blob.sentiment.polarity
    
    def _sentiment_keywords(self, text: str) -> float:
        """Sentimento baseado em keywords"""
        text_lower = text.lower()
        
        positive_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
        negative_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        # Score de -1 a 1
        score = (positive_count - negative_count) / total
        return score
    
    def _score_to_label(self, score: float) -> SentimentLabel:
        """Converte score em label"""
        if score >= 0.5:
            return SentimentLabel.VERY_POSITIVE
        elif score >= 0.1:
            return SentimentLabel.POSITIVE
        elif score <= -0.5:
            return SentimentLabel.VERY_NEGATIVE
        elif score <= -0.1:
            return SentimentLabel.NEGATIVE
        else:
            return SentimentLabel.NEUTRAL
    
    # ========================================================================
    # CLASSIFICAÇÃO DE IMPACTO
    # ========================================================================
    
    def _classify_impact(self, article: NewsArticle):
        """Classifica nível de impacto da notícia"""
        text = (article.title + ' ' + article.content).lower()
        
        # Procura keywords críticas
        critical_found = [kw for kw in CRITICAL_KEYWORDS if kw in text]
        if critical_found:
            article.impact_level = NewsImpact.CRITICAL
            article.keywords_found = critical_found
            self.critical_events_detected += 1
            logger.warning(f"🚨 Evento CRÍTICO detectado: {article.title}")
            return
        
        # Procura keywords de alto impacto
        high_impact_found = [kw for kw in HIGH_IMPACT_KEYWORDS if kw in text]
        if high_impact_found:
            article.impact_level = NewsImpact.HIGH
            article.keywords_found = high_impact_found
            return
        
        # Sentimento muito negativo = médio impacto
        if article.sentiment in [SentimentLabel.VERY_NEGATIVE, SentimentLabel.VERY_POSITIVE]:
            article.impact_level = NewsImpact.MEDIUM
            return
        
        # Resto = baixo impacto
        article.impact_level = NewsImpact.LOW
    
    # ========================================================================
    # FILTRO DE TRADING
    # ========================================================================
    
    def should_trade(
        self,
        symbol: str,
        hours_back: int = 2,
        min_sentiment_score: float = -0.3
    ) -> NewsFilter:
        """
        Determina se deve operar baseado em notícias recentes
        
        Args:
            symbol: Símbolo a verificar (ex: "BTCUSDT")
            hours_back: Horas para considerar
            min_sentiment_score: Score mínimo de sentimento
        
        Returns:
            NewsFilter com decisão e razão
        """
        # Busca notícias
        symbols = [symbol.replace("USDT", "").upper()]
        news = self.fetch_news(symbols, hours_back=hours_back)
        
        # Filtra notícias relevantes
        relevant_news = [
            n for n in news
            if any(s in n.symbols for s in symbols)
        ]
        
        if not relevant_news:
            return NewsFilter(
                should_trade=True,
                reason="Sem notícias recentes",
                recent_news=[],
                overall_sentiment=SentimentLabel.NEUTRAL,
                sentiment_score=0.0,
                critical_events=[]
            )
        
        # Verifica eventos críticos
        critical_events = [
            n for n in relevant_news
            if n.impact_level == NewsImpact.CRITICAL
        ]
        
        if critical_events:
            return NewsFilter(
                should_trade=False,
                reason=f"Evento crítico detectado: {critical_events[0].title}",
                recent_news=relevant_news,
                overall_sentiment=SentimentLabel.VERY_NEGATIVE,
                sentiment_score=-1.0,
                critical_events=critical_events
            )
        
        # Calcula sentimento geral
        if self.enable_sentiment_analysis:
            avg_sentiment = np.mean([n.sentiment_score for n in relevant_news])
            overall_sentiment = self._score_to_label(avg_sentiment)
        else:
            avg_sentiment = 0.0
            overall_sentiment = SentimentLabel.NEUTRAL
        
        # Verifica sentimento mínimo
        if avg_sentiment < min_sentiment_score:
            return NewsFilter(
                should_trade=False,
                reason=f"Sentimento muito negativo: {avg_sentiment:.2f}",
                recent_news=relevant_news,
                overall_sentiment=overall_sentiment,
                sentiment_score=avg_sentiment,
                critical_events=[]
            )
        
        # Verifica múltiplas notícias de alto impacto negativo
        high_impact_negative = [
            n for n in relevant_news
            if n.impact_level == NewsImpact.HIGH and 
            n.sentiment in [SentimentLabel.NEGATIVE, SentimentLabel.VERY_NEGATIVE]
        ]
        
        if len(high_impact_negative) >= 2:
            return NewsFilter(
                should_trade=False,
                reason="Múltiplos eventos negativos de alto impacto",
                recent_news=relevant_news,
                overall_sentiment=overall_sentiment,
                sentiment_score=avg_sentiment,
                critical_events=[]
            )
        
        # OK para operar
        return NewsFilter(
            should_trade=True,
            reason=f"Sem eventos bloqueantes (sentiment: {avg_sentiment:.2f})",
            recent_news=relevant_news,
            overall_sentiment=overall_sentiment,
            sentiment_score=avg_sentiment,
            critical_events=[]
        )
    
    # ========================================================================
    # UTILITÁRIOS
    # ========================================================================
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extrai símbolos de criptomoedas do texto"""
        symbols = []
        
        # Lista de símbolos conhecidos
        known_symbols = {
            'BTC': ['bitcoin', 'btc'],
            'ETH': ['ethereum', 'eth', 'ether'],
            'SOL': ['solana', 'sol'],
            'BNB': ['binance', 'bnb'],
            'XRP': ['ripple', 'xrp'],
            'ADA': ['cardano', 'ada'],
            'DOGE': ['dogecoin', 'doge'],
            'MATIC': ['polygon', 'matic'],
            'DOT': ['polkadot', 'dot'],
            'AVAX': ['avalanche', 'avax']
        }
        
        text_lower = text.lower()
        
        for symbol, keywords in known_symbols.items():
            if any(kw in text_lower for kw in keywords):
                symbols.append(symbol)
        
        return symbols
    
    def _deduplicate_news(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove notícias duplicadas por URL"""
        seen_urls = set()
        unique = []
        
        for article in articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique.append(article)
        
        return unique
    
    def _get_cached_news(
        self,
        symbols: List[str],
        hours_back: int
    ) -> Optional[List[NewsArticle]]:
        """Retorna notícias do cache se válidas"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        cached = [
            n for n in self.news_cache
            if n.published_at >= cutoff_time and
            any(s in n.symbols for s in symbols)
        ]
        
        return cached if cached else None
    
    def _rate_limit(self):
        """Aplica rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()
    
    def get_statistics(self) -> Dict:
        """Retorna estatísticas do filtro"""
        return {
            'total_news_fetched': self.total_news_fetched,
            'critical_events_detected': self.critical_events_detected,
            'cache_size': len(self.news_cache),
            'sentiment_engine': self.sentiment_engine,
            'sentiment_enabled': self.enable_sentiment_analysis
        }


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    """
    Exemplo de uso do News Sentiment Filter
    """
    import numpy as np
    from loguru import logger
    
    logger.add("logs/news_sentiment.log", rotation="1 day")
    
    # Inicializa (pode usar com ou sem API keys)
    news_filter = NewsSentimentFilter(
        # cryptopanic_api_key="your_key_here",  # Opcional
        # newsapi_key="your_key_here",          # Opcional
        enable_sentiment_analysis=True,
        sentiment_engine="keywords"  # Funciona sem libs externas
    )
    
    # Busca notícias
    logger.info("Buscando notícias...")
    news = news_filter.fetch_news(["BTC", "ETH"], hours_back=6)
    
    logger.info(f"Encontradas {len(news)} notícias")
    
    # Mostra notícias
    for article in news[:5]:
        logger.info(f"\n📰 {article.title}")
        logger.info(f"   Fonte: {article.source}")
        logger.info(f"   Sentimento: {article.sentiment.value if article.sentiment else 'N/A'}")
        logger.info(f"   Impacto: {article.impact_level.value if article.impact_level else 'N/A'}")
        logger.info(f"   URL: {article.url}")
    
    # Verifica se deve operar
    result = news_filter.should_trade("BTCUSDT", hours_back=2)
    
    logger.info(f"\n🎯 Decisão de Trading:")
    logger.info(f"   Operar: {result.should_trade}")
    logger.info(f"   Razão: {result.reason}")
    logger.info(f"   Sentimento Geral: {result.overall_sentiment.value}")
    logger.info(f"   Score: {result.sentiment_score:.2f}")
    
    # Estatísticas
    stats = news_filter.get_statistics()
    logger.info(f"\n📊 Estatísticas: {json.dumps(stats, indent=2)}")