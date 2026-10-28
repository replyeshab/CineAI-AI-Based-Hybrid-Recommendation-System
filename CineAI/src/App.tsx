import React, { useState, useMemo } from 'react';
import { AuthProvider, useAuth } from './context/AuthContext';
import { Navbar } from './components/Navbar';
import { HeroBanner } from './components/HeroBanner';
import { MovieRow } from './components/MovieRow';
import { MovieModal } from './components/MovieModal';
import { EngineInspector } from './components/EngineInspector';
import { ProfileView } from './components/ProfileView';
import { AuthModal } from './components/AuthModal';
import { MovieCard } from './components/MovieCard';
import { MOVIELENS_DATASET } from './data/movies';
import { ScoredMovie, Movie } from './types';
import {
  recommendMoviesHybrid,
  recommendPopular,
  calculateImdbScore,
  recommendUserToUserCollaborative,
  findKNearestNeighbors
} from './services/recommender';
import { useApiRecommendations, useSemanticSearch } from './services/cineaiApi';
import { api as sessionApi, getToken } from './services/authApi';
import { OtpDialog } from './components/OtpDialog';
import { AccountSettings } from './components/AccountSettings';
import { NetflixAuthPage } from './components/NetflixAuthPage';
import { ContinueWatchingRow } from './components/ContinueWatchingRow';
import { DeleteAccountModal } from './components/DeleteAccountModal';
import { WatchlistToolbar, WatchlistSortOption } from './components/WatchlistToolbar';
import { AgenticAiRecommender } from './components/AgenticAiRecommender';
import { Film, Sparkles, SlidersHorizontal, Search, Star, Heart, Users, Bookmark, Bot, Compass } from 'lucide-react';

const MainContent: React.FC = () => {
  const { user, hasGuestEntered, setHasGuestEntered, toggleWatchlist } = useAuth();
  const [currentTab, setCurrentTab] = useState<'browse' | 'agent' | 'top-rated' | 'watchlist' | 'inspector' | 'profile' | 'login'>('browse');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchMode, setSearchMode] = useState<'keyword' | 'semantic'>('keyword');
  const [selectedMovieForModal, setSelectedMovieForModal] = useState<ScoredMovie | null>(null);
  const [isGlobalDeleteModalOpen, setIsGlobalDeleteModalOpen] = useState(false);

  // Watchlist Filter and Sort States
  const [watchlistSearch, setWatchlistSearch] = useState('');
  const [watchlistGenre, setWatchlistGenre] = useState('all');
  const [watchlistSort, setWatchlistSort] = useState<WatchlistSortOption>('score-desc');

  // Compute Hybrid Recommendations based on active user state and custom weights
  const localRecommendations = useMemo(() => recommendMoviesHybrid(user, { topN: 30, filterWatched: false }), [user]);
  // Trained Python model (with TMDB posters) when VITE_CINEAI_API_URL is set; local engine otherwise
  const { recs: hybridRecommendations, status: engineStatus } = useApiRecommendations(user, localRecommendations);

  // Featured Marquee film: top recommended movie
  const heroMovie = useMemo(() => {
    return hybridRecommendations[0] || null;
  }, [hybridRecommendations]);

  // Content-based row: "Because You Liked Inception (or user's top rated film)"
  const referenceFilm = useMemo(() => {
    if (user && Object.keys(user.ratings || {}).length > 0) {
      const topId = Object.entries(user.ratings || {}).sort((a, b) => b[1] - a[1])[0]?.[0];
      return MOVIELENS_DATASET.find(m => m.movieId === Number(topId)) || MOVIELENS_DATASET[5];
    }
    return MOVIELENS_DATASET[5]; // Inception
  }, [user]);

  const contentSimilarRecommendations = useMemo(() => {
    return recommendMoviesHybrid(user, {
      referenceMovie: referenceFilm,
      topN: 15,
      filterWatched: false,
      customWeights: {
        content_score: 0.70,
        genre_score: 0.20,
        popularity_score: 0.10,
        collab_score: 0.00
      }
    });
  }, [user, referenceFilm]);

  // User-to-User Collaborative Filtering (k-Nearest Neighbors CF)
  const peopleLikeYouRecommendations = useMemo(() => {
    return recommendUserToUserCollaborative(user, MOVIELENS_DATASET, 4, 15);
  }, [user]);

  const nearestNeighbors = useMemo(() => {
    return findKNearestNeighbors(user, 3);
  }, [user]);

  // Bayesian Popularity row
  const popularRecommendations = useMemo(() => {
    const popularRaw = recommendPopular(MOVIELENS_DATASET, 15);
    return popularRaw.map(m => {
      const scored = hybridRecommendations.find(s => s.movie.movieId === m.movieId);
      return scored || {
        movie: m,
        final_score: 0.85,
        match_percentage: 88,
        sub_scores: {
          collab_score: 0.5,
          content_score: 0.5,
          popularity_score: 0.9,
          genre_score: 0.5,
          rating_score: 0.8,
          rating_count_score: 0.9,
          recency_score: 0.5
        },
        weights_used: {
          collab_score: 0,
          content_score: 0,
          popularity_score: 1,
          genre_score: 0,
          rating_score: 0,
          rating_count_score: 0,
          recency_score: 0
        },
        explanation: `Ranked in top global tier based on IMDb Bayesian weighted score (${(m.weighted_rating || 4.2).toFixed(2)}) and ${m.rating_count.toLocaleString()} verified ratings.`,
        contributing_factors: []
      };
    });
  }, [hybridRecommendations]);

  // Sci-Fi Spotlight row
  const sciFiRecommendations = useMemo(() => {
    return hybridRecommendations.filter(s => s.movie.genres.includes('Sci-Fi')).slice(0, 15);
  }, [hybridRecommendations]);

  // High Recency (Modern Cinema) row
  const modernRecommendations = useMemo(() => {
    return [...hybridRecommendations]
      .sort((a, b) => b.movie.year - a.movie.year)
      .slice(0, 15);
  }, [hybridRecommendations]);

  // Search Results filtering (supports both keyword token matching and AI semantic concept matching)
  const localSearchResults = useMemo(() => {
    if (!searchQuery.trim()) return [];
    const q = searchQuery.toLowerCase().trim();

    if (searchMode === 'semantic') {
      const qTokens = q.split(/\s+/).filter(Boolean);
      const scored = hybridRecommendations.map(s => {
        let relevance = 0;
        const text = (
          s.movie.clean_title + ' ' +
          s.movie.genres + ' ' +
          s.movie.synopsis + ' ' +
          s.movie.director + ' ' +
          s.movie.tfidf_keywords.join(' ')
        ).toLowerCase();

        qTokens.forEach(token => {
          if (text.includes(token)) relevance += 16;
        });

        // Conceptual thematic mapping
        if ((q.includes('space') || q.includes('cosmos') || q.includes('star')) && s.movie.genres.includes('Sci-Fi')) relevance += 22;
        if ((q.includes('dark') || q.includes('neon') || q.includes('noir') || q.includes('detective')) && (s.movie.genres.includes('Crime') || s.movie.genres.includes('Mystery') || s.movie.genres.includes('Thriller'))) relevance += 22;
        if ((q.includes('mind') || q.includes('dream') || q.includes('reality') || q.includes('twist')) && (s.movie.clean_title.includes('Inception') || s.movie.clean_title.includes('Matrix') || s.movie.clean_title.includes('Fight Club') || s.movie.clean_title.includes('Prestige'))) relevance += 26;
        if ((q.includes('warm') || q.includes('comfort') || q.includes('animated') || q.includes('anime')) && s.movie.genres.includes('Animation')) relevance += 25;
        if ((q.includes('family') || q.includes('mafia') || q.includes('loyalty')) && s.movie.clean_title.includes('Godfather')) relevance += 28;

        const finalMatch = Math.min(99, Math.max(68, relevance + s.match_percentage * 0.4));
        return {
          ...s,
          match_percentage: Math.round(finalMatch),
          explanation: `Semantic match (${Math.round(finalMatch)}%): Resonates with thematic intent "${searchQuery}" based on narrative motifs and ${s.movie.genres} direction.`
        };
      });

      return scored.filter(s => s.match_percentage > 72).sort((a, b) => b.match_percentage - a.match_percentage);
    }

    // Default Keyword Filter
    return hybridRecommendations.filter(
      s =>
        s.movie.clean_title.toLowerCase().includes(q) ||
        s.movie.genres.toLowerCase().includes(q) ||
        s.movie.director.toLowerCase().includes(q) ||
        s.movie.synopsis.toLowerCase().includes(q)
    );
  }, [searchQuery, searchMode, hybridRecommendations]);

  // Real description search (embeddings + personal re-ranking) when the server is connected; local keyword search otherwise
  const { results: semanticResults, loading: searchLoading } = useSemanticSearch(searchQuery, searchMode === 'semantic', user);
  const searchResults = searchMode === 'semantic' && semanticResults ? semanticResults : localSearchResults;
  const usingServerSearch = searchMode === 'semantic' && !!semanticResults;
  // picking a result of a description search is a (weak) positive signal for the model
  const openFromSearch = (m: ScoredMovie) => {
    setSelectedMovieForModal(m);
    if (usingServerSearch && getToken()) sessionApi('/me/events', 'POST', { movieId: m.movie.movieId, event: 'search_pick' }).catch(() => {});
  };

  // Top Rated list
  const topRatedMovies = useMemo(() => {
    const withBayesian = calculateImdbScore(MOVIELENS_DATASET);
    const sorted = [...withBayesian].sort((a, b) => (b.weighted_rating || 0) - (a.weighted_rating || 0));
    return sorted.map((m, idx) => {
      const match = hybridRecommendations.find(s => s.movie.movieId === m.movieId);
      return {
        scored: match || {
          movie: m,
          final_score: 0.9,
          match_percentage: 95,
          sub_scores: {
            collab_score: 0.5,
            content_score: 0.5,
            popularity_score: 1.0,
            genre_score: 0.5,
            rating_score: 0.9,
            rating_count_score: 0.9,
            recency_score: 0.5
          },
          weights_used: {
            collab_score: 0.2,
            content_score: 0.3,
            popularity_score: 0.5,
            genre_score: 0,
            rating_score: 0,
            rating_count_score: 0,
            recency_score: 0
          },
          explanation: `IMDb Bayesian consensus score of ${(m.weighted_rating || 0).toFixed(3)} across verified MovieLens interactions.`,
          contributing_factors: []
        },
        rank: idx + 1
      };
    });
  }, [hybridRecommendations]);

  // Watchlist items
  const watchlistScored = useMemo(() => {
    if (!user) return [];
    return (user.watchlist || [])
      .map(id => hybridRecommendations.find(s => s.movie.movieId === id))
      .filter((s): s is ScoredMovie => Boolean(s));
  }, [user, hybridRecommendations]);

  // Extract available genres with counts from active watchlist
  const availableWatchlistGenres = useMemo(() => {
    const counts: Record<string, number> = {};
    watchlistScored.forEach(s => {
      s.movie.genres.split('|').forEach(g => {
        const trimmed = g.trim();
        if (trimmed) {
          counts[trimmed] = (counts[trimmed] || 0) + 1;
        }
      });
    });
    return Object.entries(counts)
      .map(([genre, count]) => ({ genre, count }))
      .sort((a, b) => b.count - a.count);
  }, [watchlistScored]);

  // Filtered and sorted watchlist items
  const filteredAndSortedWatchlist = useMemo(() => {
    let result = [...watchlistScored];

    // 1. Filter by Genre
    if (watchlistGenre !== 'all') {
      result = result.filter(s => s.movie.genres.split('|').includes(watchlistGenre));
    }

    // 2. Filter by Search Query
    if (watchlistSearch.trim()) {
      const q = watchlistSearch.toLowerCase().trim();
      result = result.filter(
        s =>
          s.movie.clean_title.toLowerCase().includes(q) ||
          s.movie.director.toLowerCase().includes(q) ||
          s.movie.synopsis.toLowerCase().includes(q) ||
          s.movie.genres.toLowerCase().includes(q)
      );
    }

    // 3. Sort by Selected Option
    result.sort((a, b) => {
      switch (watchlistSort) {
        case 'score-desc':
          return (b.match_percentage || 0) - (a.match_percentage || 0);
        case 'score-asc':
          return (a.match_percentage || 0) - (b.match_percentage || 0);
        case 'year-desc':
          return b.movie.year - a.movie.year;
        case 'year-asc':
          return a.movie.year - b.movie.year;
        case 'rating-desc':
          return (b.movie.weighted_rating || b.movie.average_rating) - (a.movie.weighted_rating || a.movie.average_rating);
        case 'title-asc':
          return a.movie.clean_title.localeCompare(b.movie.clean_title);
        case 'title-desc':
          return b.movie.clean_title.localeCompare(a.movie.clean_title);
        default:
          return 0;
      }
    });

    return result;
  }, [watchlistScored, watchlistGenre, watchlistSearch, watchlistSort]);

  const handleAddStarterWatchlist = () => {
    // Benchmark titles across genres: Inception (Sci-Fi), Pulp Fiction (Crime), The Dark Knight (Action), Spirited Away (Animation), The Godfather (Drama)
    const starterIds = [79132, 296, 58559, 5618, 858];
    starterIds.forEach(id => {
      if (!user?.watchlist?.includes(id)) {
        toggleWatchlist(id);
      }
    });
  };

  const handlePlayMovie = (movie: Movie, progressSeconds: number) => {
    const scored = hybridRecommendations.find(s => s.movie.movieId === movie.movieId) || {
      movie,
      final_score: 0.92,
      match_percentage: 95,
      sub_scores: {
        collab_score: 0.8,
        content_score: 0.8,
        popularity_score: 0.9,
        genre_score: 0.8,
        rating_score: movie.average_rating / 5,
        rating_count_score: 0.8,
        recency_score: 0.5
      },
      weights_used: {
        collab_score: 0.2,
        content_score: 0.3,
        popularity_score: 0.2,
        genre_score: 0.3,
        rating_score: 0,
        rating_count_score: 0,
        recency_score: 0
      },
      explanation: `Resuming from ${Math.floor(progressSeconds / 60)} minutes.`,
      contributing_factors: []
    };
    setSelectedMovieForModal(scored);
  };

  // If user is not logged in and has not chosen to explore as guest, or unauthenticated user clicked Login
  if (!user && (!hasGuestEntered || currentTab === 'login')) {
    return (
      <NetflixAuthPage
        initialMode={currentTab === 'login' ? 'login' : 'landing'}
        onEnterAsGuest={() => {
          setHasGuestEntered(true);
          setCurrentTab('browse');
        }}
        onLoginSuccess={() => {
          setHasGuestEntered(true);
          setCurrentTab('browse');
        }}
      />
    );
  }

  return (
    <div className="min-h-screen bg-[#08090C] text-neutral-100 flex flex-col selection:bg-red-600 selection:text-white">
      {/* Top Bar Contract Navigation */}
      <OtpDialog />
      <AccountSettings />
      <div className="fixed bottom-3 right-3 z-[60] rounded-full bg-black/80 border border-white/10 px-3 py-1 text-[11px] text-white/80 pointer-events-none">
        <span className={`inline-block w-2 h-2 rounded-full mr-1.5 ${engineStatus === 'online' ? 'bg-green-500' : engineStatus === 'loading' ? 'bg-yellow-400 animate-pulse' : 'bg-white/40'}`} />
        {engineStatus === 'online' ? 'Trained model · live' : engineStatus === 'loading' ? 'Contacting model…' : 'Local engine'}
      </div>
      <Navbar
        currentTab={currentTab}
        onSelectTab={tab => {
          setCurrentTab(tab);
          setSearchQuery('');
          window.scrollTo({ top: 0, behavior: 'smooth' });
        }}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        searchMode={searchMode}
        onSearchModeChange={setSearchMode}
        onOpenDeleteModal={() => setIsGlobalDeleteModalOpen(true)}
      />

      {/* Main Content Viewport */}
      <main className="flex-1">
        {/* Search View Override */}
        {searchQuery.trim() ? (
          <div className="w-full max-w-7xl mx-auto px-6 lg:px-12 py-10 space-y-6">
            <div className="border-b border-white/[0.08] pb-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
                  <Search className="w-5 h-5 text-red-500" />
                  <span>
                    {searchMode === 'semantic' ? 'AI Semantic Search' : 'Search Results'}{searchLoading ? ' · thinking…' : ''} for "{searchQuery}"
                  </span>
                </h1>
                <p className="text-xs text-neutral-400 mt-1">
                  Found {searchResults.length} {searchMode === 'semantic' ? 'thematically ranked' : 'matching'} films {usingServerSearch ? (user && Object.keys(user.ratings || {}).length ? 'that fit your description, tuned to your taste' : 'that fit your description') : 'in catalog'}
                </p>
              </div>

              {/* Mode Switcher Pill */}
              <div className="flex items-center gap-1.5 p-1 rounded-xl bg-white/5 border border-white/10 text-xs self-start sm:self-auto">
                <button
                  type="button"
                  onClick={() => setSearchMode('keyword')}
                  className={`px-3 py-1.5 rounded-lg font-medium transition-all cursor-pointer ${
                    searchMode === 'keyword'
                      ? 'bg-red-600 text-white font-bold shadow-md'
                      : 'text-neutral-400 hover:text-white'
                  }`}
                >
                  Keyword Match
                </button>
                <button
                  type="button"
                  onClick={() => setSearchMode('semantic')}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-all cursor-pointer ${
                    searchMode === 'semantic'
                      ? 'bg-amber-500 text-neutral-950 font-bold shadow-md'
                      : 'text-neutral-400 hover:text-white'
                  }`}
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>AI Semantic</span>
                </button>
              </div>
            </div>

            {/* Quick Semantic Mood Suggestion Chips */}
            {searchMode === 'semantic' && (
              <div className="flex items-center gap-2 overflow-x-auto pb-1 scrollbar-none text-xs">
                <span className="text-[11px] font-mono text-amber-400 shrink-0 font-semibold flex items-center gap-1">
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Semantic Themes:</span>
                </span>
                {[
                  'Space isolation & cosmos',
                  'Dark neon noir mystery',
                  'Mind-bending reality twist',
                  'Warm animated comfort',
                  'High stakes tactical heist',
                  'Epic family saga'
                ].map(prompt => (
                  <button
                    key={prompt}
                    type="button"
                    onClick={() => setSearchQuery(prompt)}
                    className={`px-3 py-1 rounded-full text-xs font-medium shrink-0 cursor-pointer transition-all ${
                      searchQuery.toLowerCase() === prompt.toLowerCase()
                        ? 'bg-amber-500 text-neutral-950 font-bold'
                        : 'bg-white/5 hover:bg-white/10 text-neutral-300 hover:text-white border border-white/10'
                    }`}
                  >
                    {prompt}
                  </button>
                ))}
                <button
                  type="button"
                  onClick={() => {
                    setSearchQuery('');
                    setCurrentTab('agent');
                  }}
                  className="ml-auto px-3 py-1 rounded-full bg-red-600/20 text-red-400 border border-red-500/30 hover:bg-red-600/30 text-xs font-semibold shrink-0 cursor-pointer flex items-center gap-1.5"
                >
                  <Bot className="w-3.5 h-3.5" />
                  <span>Open Full CineAgent AI</span>
                </button>
              </div>
            )}

            {searchResults.length > 0 ? (
              <div className="stagger grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6">
                {searchResults.map(scored => (
                  <MovieCard
                    key={scored.movie.movieId}
                    scoredMovie={scored}
                    onOpenDetails={openFromSearch}
                  />
                ))}
              </div>
            ) : (
              <div className="text-center py-20 space-y-3">
                <p className="text-neutral-400 text-sm">
                  No movies found matching "{searchQuery}". Try searching for "Sci-Fi", "Nolan", or "Crime".
                </p>
              </div>
            )}
          </div>
        ) : (
          <>
            {/* TAB: BROWSE / DISCOVERY */}
            {currentTab === 'browse' && (
              <div className="space-y-6">
                {/* Hero Featured Billboard */}
                {heroMovie && (
                  <HeroBanner
                    scoredMovie={heroMovie}
                    onOpenDetails={setSelectedMovieForModal}
                    onOpenInspector={() => setCurrentTab('inspector')}
                  />
                )}

                {/* Algorithmic Streaming Carousels */}
                <div className="relative z-20 -mt-8 sm:-mt-14 space-y-2 pb-16">
                  {/* Continue Watching Row (Tracks movie progress & timestamps from user watch history) */}
                  <ContinueWatchingRow
                    onOpenDetails={setSelectedMovieForModal}
                    onPlayMovie={handlePlayMovie}
                  />

                  {/* Top Picks for You (Personalized Hybrid SVD + Content + Genre) */}
                  <MovieRow
                    title="Top Picks for You"
                    subtitle="Ranked with CineAI Hybrid Dynamic Weights based on your taste vector"
                    badge="Hybrid Rank"
                    scoredMovies={hybridRecommendations.slice(0, 15)}
                    onOpenDetails={setSelectedMovieForModal}
                  />

                  {/* People Like You (User-to-User Collaborative Filtering k-NN) */}
                  <MovieRow
                    title="People Like You"
                    subtitle={`Discovered via k-NN peer clustering across your nearest tastemakers: ${nearestNeighbors.map(n => `${n.name} (${Math.round(n.similarity * 100)}%)`).join(', ')}`}
                    badge="k-NN Peers"
                    scoredMovies={peopleLikeYouRecommendations}
                    onOpenDetails={setSelectedMovieForModal}
                  />

                  {/* Because You Liked [Reference Film] (TF-IDF Content Similarity) */}
                  <MovieRow
                    title={`Because You Liked ${referenceFilm.clean_title}`}
                    subtitle="TF-IDF keyword correlation, genre overlap, and directorial affinity"
                    badge="Content Match"
                    scoredMovies={contentSimilarRecommendations}
                    onOpenDetails={setSelectedMovieForModal}
                  />

                  {/* Trending on CineAI (Bayesian IMDb Score) */}
                  <MovieRow
                    title="Trending on CineAI"
                    subtitle="Ranked by IMDb Bayesian weighted rating formula (v / (v + m)) * R + (m / (v + m)) * C"
                    badge="Bayesian Top"
                    scoredMovies={popularRecommendations}
                    onOpenDetails={setSelectedMovieForModal}
                    showRanks={true}
                  />

                  {/* Sci-Fi & Speculative Fiction */}
                  <MovieRow
                    title="Sci-Fi & High Concept Cinema"
                    subtitle="Curated titles exploring existential futures, artificial minds, and cosmic voyages"
                    badge="Genre Focus"
                    scoredMovies={sciFiRecommendations}
                    onOpenDetails={setSelectedMovieForModal}
                  />

                  {/* Modern Cinema & Recency */}
                  <MovieRow
                    title="Modern Cinema Horizons"
                    subtitle="Recent benchmark releases scored with recency decay factors"
                    badge="High Recency"
                    scoredMovies={modernRecommendations}
                    onOpenDetails={setSelectedMovieForModal}
                  />
                </div>
              </div>
            )}

            {/* TAB: TOP RATED */}
            {currentTab === 'top-rated' && (
              <div className="w-full max-w-7xl mx-auto px-6 lg:px-12 py-10 space-y-8">
                <div className="border-b border-white/[0.08] pb-4">
                  <div className="flex items-center gap-2 text-xs font-mono text-amber-400 mb-1">
                    <Star className="w-4 h-4 fill-amber-400" />
                    <span className="uppercase tracking-widest">IMDb Bayesian Ranking</span>
                  </div>
                  <h1 className="text-3xl font-extrabold tracking-tight text-white">
                    MovieLens All-Time Top Rated
                  </h1>
                  <p className="text-xs text-neutral-400 mt-1 max-w-2xl">
                    Ranked by Bayesian weighted average rating (m = 90th percentile threshold) preventing title anomalies with low vote counts from distorting the leaderboard.
                  </p>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6">
                  {topRatedMovies.map(({ scored, rank }) => (
                    <MovieCard
                      key={scored.movie.movieId}
                      scoredMovie={scored}
                      onOpenDetails={setSelectedMovieForModal}
                      showRank={rank}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* TAB: WATCHLIST */}
            {currentTab === 'watchlist' && (
              <div className="w-full max-w-7xl mx-auto px-6 lg:px-12 py-10 space-y-6">
                <div className="border-b border-white/[0.08] pb-4 flex flex-col sm:flex-row sm:items-end justify-between gap-4">
                  <div>
                    <h1 className="text-3xl font-extrabold tracking-tight text-white flex items-center gap-2.5">
                      <Bookmark className="w-7 h-7 text-red-500 fill-red-500/20" />
                      <span>Your Watchlist</span>
                    </h1>
                    <p className="text-xs text-neutral-400 mt-1">
                      {watchlistScored.length} films saved for discovery across your devices.
                    </p>
                  </div>

                  {watchlistScored.length > 0 && (
                    <div className="flex items-center gap-2">
                      <button
                        onClick={handleAddStarterWatchlist}
                        className="px-3.5 py-1.5 rounded-xl bg-white/5 hover:bg-white/10 text-neutral-300 hover:text-white border border-white/10 text-xs font-medium transition-all cursor-pointer"
                      >
                        + Add Recommended Films
                      </button>
                    </div>
                  )}
                </div>

                {watchlistScored.length > 0 ? (
                  <>
                    {/* Dedicated Sticky Filter & Sort Toolbar */}
                    <WatchlistToolbar
                      totalCount={watchlistScored.length}
                      filteredCount={filteredAndSortedWatchlist.length}
                      searchQuery={watchlistSearch}
                      onSearchChange={setWatchlistSearch}
                      selectedGenre={watchlistGenre}
                      onSelectGenre={setWatchlistGenre}
                      availableGenres={availableWatchlistGenres}
                      sortOption={watchlistSort}
                      onSortChange={setWatchlistSort}
                      onResetFilters={() => {
                        setWatchlistSearch('');
                        setWatchlistGenre('all');
                        setWatchlistSort('score-desc');
                      }}
                      hasActiveFilters={
                        watchlistGenre !== 'all' ||
                        Boolean(watchlistSearch.trim()) ||
                        watchlistSort !== 'score-desc'
                      }
                    />

                    {/* Movie Grid or Filtered Zero State */}
                    {filteredAndSortedWatchlist.length > 0 ? (
                      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6 pt-2">
                        {filteredAndSortedWatchlist.map(scored => (
                          <MovieCard
                            key={scored.movie.movieId}
                            scoredMovie={scored}
                            onOpenDetails={setSelectedMovieForModal}
                          />
                        ))}
                      </div>
                    ) : (
                      <div className="text-center py-20 bg-white/[0.02] border border-white/5 rounded-2xl p-8 space-y-4">
                        <p className="text-neutral-300 text-sm font-semibold">
                          No saved films match the filter criteria.
                        </p>
                        <p className="text-xs text-neutral-500 max-w-sm mx-auto">
                          Try selecting a different genre or clearing your search keywords to view your saved library.
                        </p>
                        <button
                          onClick={() => {
                            setWatchlistSearch('');
                            setWatchlistGenre('all');
                            setWatchlistSort('score-desc');
                          }}
                          className="px-4 py-2 rounded-xl bg-red-600 hover:bg-red-700 text-white text-xs font-bold transition-all shadow-lg cursor-pointer"
                        >
                          Clear Applied Filters
                        </button>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="text-center py-24 bg-white/[0.02] border border-white/5 rounded-3xl p-10 space-y-5">
                    <div className="w-16 h-16 rounded-2xl bg-red-600/10 border border-red-500/20 text-red-500 flex items-center justify-center mx-auto shadow-xl">
                      <Bookmark className="w-8 h-8" />
                    </div>
                    <div className="space-y-1.5 max-w-md mx-auto">
                      <h2 className="text-lg font-bold text-white">Your Watchlist is empty</h2>
                      <p className="text-neutral-400 text-xs">
                        Save films while browsing to build your personalized discovery queue, or load recommended starter films immediately.
                      </p>
                    </div>
                    <div className="flex flex-wrap items-center justify-center gap-3 pt-2">
                      <button
                        onClick={handleAddStarterWatchlist}
                        className="px-5 py-2.5 rounded-xl bg-red-600 hover:bg-red-700 text-white text-xs font-bold transition-all shadow-lg shadow-red-950/50 cursor-pointer active:scale-95"
                      >
                        + Add 5 Top Recommended Starters
                      </button>
                      <button
                        onClick={() => setCurrentTab('browse')}
                        className="px-5 py-2.5 rounded-xl bg-white/10 hover:bg-white/15 text-neutral-200 hover:text-white border border-white/15 text-xs font-semibold transition-all cursor-pointer"
                      >
                        Browse Catalog
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* TAB: CINEAGENT AI (AGENTIC RECOMMENDER) */}
            {currentTab === 'agent' && (
              <AgenticAiRecommender
                onOpenDetails={setSelectedMovieForModal}
                onPlayMovie={handlePlayMovie}
              />
            )}

            {/* TAB: ENGINE INSPECTOR */}
            {currentTab === 'inspector' && (
              <EngineInspector onOpenDetails={setSelectedMovieForModal} />
            )}

            {/* TAB: PROFILE */}
            {currentTab === 'profile' && (
              <ProfileView
                onOpenDetails={setSelectedMovieForModal}
                onOpenInspector={() => setCurrentTab('inspector')}
                onNavigateToLogin={() => setCurrentTab('login')}
                onOpenDeleteModal={() => setIsGlobalDeleteModalOpen(true)}
              />
            )}
          </>
        )}
      </main>

      {/* Global Movie Detail Modal */}
      {selectedMovieForModal && (
        <MovieModal
          scoredMovie={selectedMovieForModal}
          onClose={() => setSelectedMovieForModal(null)}
          onOpenInspector={() => {
            setSelectedMovieForModal(null);
            setCurrentTab('inspector');
          }}
        />
      )}

      {/* Global Custom Auth Modal (Login / Register / Taste Calibration) */}
      <AuthModal />

      {/* Global Delete Account Confirmation Modal */}
      <DeleteAccountModal
        isOpen={isGlobalDeleteModalOpen}
        onClose={() => setIsGlobalDeleteModalOpen(false)}
        onDeleted={() => {
          setIsGlobalDeleteModalOpen(false);
          setCurrentTab('login');
        }}
      />

      {/* Quiet Footer */}
      <footer className="border-t border-white/[0.07] bg-[#06070a] px-6 lg:px-12 py-12 text-neutral-400 text-xs mt-20">
        <div className="w-full max-w-7xl mx-auto flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="flex items-center justify-center w-6 h-6 rounded-md bg-red-600 text-white">
                <Film className="w-3.5 h-3.5" />
              </span>
              <span className="font-extrabold text-sm tracking-tight text-white">
                CINE<span className="text-red-600">AI</span>
              </span>
            </div>
            <p className="text-[11px] text-neutral-500 max-w-md">
              Cinema-grade recommendation platform powered by the MovieLens 32M corpus. Implements Bayesian IMDb popularity scoring, SVD collaborative filtering, and TF-IDF content similarity.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-6 text-neutral-400 font-medium">
            <button
              onClick={() => setCurrentTab('browse')}
              className="hover:text-white transition-colors cursor-pointer"
            >
              Browse Catalog
            </button>
            <button
              onClick={() => setCurrentTab('agent')}
              className="hover:text-amber-300 text-amber-400 transition-colors cursor-pointer font-bold flex items-center gap-1.5"
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>CineAgent AI</span>
            </button>
            {!user ? (
              <button
                onClick={() => setCurrentTab('login')}
                className="hover:text-red-400 text-white transition-colors cursor-pointer font-semibold flex items-center gap-1"
              >
                <span className="text-red-500">●</span>
                <span>Sign In</span>
              </button>
            ) : (
              <button
                onClick={() => setCurrentTab('profile')}
                className="text-white hover:text-red-400 transition-colors cursor-pointer font-semibold flex items-center gap-1.5"
              >
                <span className="w-2 h-2 rounded-full bg-emerald-500" />
                <span>{user.username}</span>
              </button>
            )}
            <button
              onClick={() => setCurrentTab('inspector')}
              className="hover:text-white transition-colors cursor-pointer"
            >
              Engine Inspector
            </button>
            <button
              onClick={() => setCurrentTab('profile')}
              className="hover:text-white transition-colors cursor-pointer"
            >
              Taste Profile
            </button>
            <span className="text-neutral-600">·</span>
            <span className="text-neutral-500 font-mono text-[10px]">
              Offline Latency: 0.0038s
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
};

class AppErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('CineAI ErrorBoundary caught:', error, errorInfo);
  }

  handleReload = () => {
    try {
      localStorage.removeItem('cineai_current_user');
      localStorage.removeItem('cineai_guest_entered');
    } catch {
      // ignore
    }
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-[#08090C] text-white flex flex-col items-center justify-center p-6 text-center font-sans">
          <div className="w-16 h-16 rounded-2xl bg-red-600/20 border border-red-500/40 text-red-500 flex items-center justify-center mb-6">
            <Film className="w-8 h-8" />
          </div>
          <h1 className="text-2xl font-bold tracking-tight mb-2">CineAI Streaming Interface Notice</h1>
          <p className="text-neutral-400 text-sm max-w-md mb-6">
            The application experienced an unexpected view interruption. Click below to restore full recommendation catalog service.
          </p>
          <button
            onClick={this.handleReload}
            className="px-6 py-3 rounded-xl bg-red-600 hover:bg-red-700 text-white font-bold text-sm transition-all shadow-lg shadow-red-950/50 cursor-pointer active:scale-95"
          >
            Restart & Restore Catalog
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  return (
    <AppErrorBoundary>
      <AuthProvider>
        <MainContent />
      </AuthProvider>
    </AppErrorBoundary>
  );
}
