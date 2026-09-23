// Browser and Node share this exact inference implementation. No dependencies.
export function validateBundle(b) {
  if (!b || b.schema !== 1 || b.method !== 'mean-unit-item-cosine-v1' ||
      !Array.isArray(b.movies) || b.movies.length < 2 || b.movies.length > 10000 ||
      !Array.isArray(b.vectors) || !Array.isArray(b.popularity) ||
      b.vectors.length !== b.movies.length || b.popularity.length !== b.movies.length)
    throw new Error('Unsupported bundle. Generate one with the project training command.');
  const d = b.vectors[0]?.length;
  if (!d || d > 256 || b.vectors.some(v => !Array.isArray(v) || v.length !== d || v.some(x => !Number.isFinite(x))) ||
      b.popularity.some(x => !Number.isFinite(x) || x < 0) ||
      b.movies.some(m => !Number.isSafeInteger(m.id) || typeof m.title !== 'string' || m.title.length > 300) ||
      new Set(b.movies.map(m => m.id)).size !== b.movies.length)
    throw new Error('Invalid catalog or embedding values.');
  return b;
}

export function recommend(bundle, favoriteIds = [], k = 10) {
  validateBundle(bundle);
  if (!Number.isInteger(k) || k < 1) throw new Error('Invalid recommendation count.');
  const selected = new Set(favoriteIds);
  const indices = bundle.movies.map((m, i) => selected.has(m.id) ? i : -1).filter(i => i >= 0);
  if (indices.length !== selected.size) throw new Error('Unknown favorite movie ID.');
  let scores;
  if (!indices.length) scores = bundle.popularity.slice();
  else {
    const unit = bundle.vectors.map(v => {
      const norm = Math.max(Math.sqrt(v.reduce((s, x) => s + x * x, 0)), 1e-12);
      return v.map(x => x / norm);
    });
    const profile = unit[0].map((_, j) => indices.reduce((s, i) => s + unit[i][j], 0) / indices.length);
    scores = unit.map(v => v.reduce((s, x, j) => s + x * profile[j], 0));
  }
  return bundle.movies.map((m, i) => ({ ...m, score: scores[i], index: i }))
    .filter(m => !selected.has(m.id))
    .sort((a, b) => b.score - a.score || a.index - b.index).slice(0, k);
}
