# Synchronicity web dashboard

Next.js scaffold for the analytics dashboard (docs/plan.md — build order: match
report → rally player → serve/scenario views → heatmaps → profiles).

> **UNVERIFIED SCAFFOLD.** This package was written on a machine **without Node.js
> installed** — it has never been through `npm install && npm run build`. Treat every
> file as a draft until a build passes. Expect at minimum pinned-version drift and
> possibly small TS/JSX fixes on first build.

## Requirements

- Node.js ≥ 20 (see `engines` in package.json)
- The FastAPI service running (default `http://localhost:8000`)

## Run

```bash
cd web
npm install
npm run dev          # http://localhost:3000
```

Point at a non-default API:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

Production build (this is the verification gate for the scaffold):

```bash
npm run build && npm run start
```

## Layout

- `app/page.tsx` — match list, graceful "API not running" / empty states
- `app/matches/[id]/page.tsx` — match report placeholder (header, score by game,
  confidence badge, rally table with clip-link placeholders)
- `components/CourtSVG.tsx` — **the** shared SVG court (geometry mirrors
  `pipeline/src/synchro_pipeline/domain/court.py`; keep in sync). All future court
  views (heatmaps, trajectories, rally player) overlay this component via its
  children slot — do not fork court drawings.
- `lib/api.ts` — typed fetch helpers; TS interfaces mirror
  `pipeline/src/synchro_pipeline/schemas/records.py`. API routes/fields may drift
  while the FastAPI service is scaffolded in parallel; reconcile here only.

## Deliberately absent (for now)

- Tailwind/shadcn/visx/hls.js (plan stack) — added once the toolchain can be
  verified; plain `globals.css` keeps the scaffold honest.
- Auth (Clerk), uploads, rally player — later phases.
