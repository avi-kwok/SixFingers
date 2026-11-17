# SixFingers web

Next.js (App Router, TypeScript) front end. One page to check an image, plus About and Privacy pages. The browser uploads straight to the API.

## Run

    cd web
    npm install
    cp .env.example .env.local   # edit as needed
    npm run dev                  # http://localhost:3000
    npm run lint
    npm run build

## Environment

| Variable | Meaning |
|---|---|
| `NEXT_PUBLIC_API_URL` | API base URL, no trailing slash. Default `http://localhost:8000`. |
| `NEXT_PUBLIC_MOCK` | `1` renders a fixture result and fake metrics without calling the API. |

The API must allow this origin via CORS.

## Layout

- `app/page.tsx` upload, preview, progress, result card, error states (429, 413, 415, network).
- `app/about`, `app/privacy` static text; About reads `GET /v1/model` and shows whatever metrics it returns.
- `lib/types.ts` mirrors `ml/common/api_models.py` by hand. Update both together.
- `lib/api.ts` upload (XHR for progress), mock fixture.
