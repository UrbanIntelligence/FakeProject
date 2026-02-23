# Fake Human Face Image Detector (Website Skeleton)

Static website skeleton for a fake-human-face detector product.

## Included in this skeleton
- Upload input (`file`) and image link input (`URL`)
- Major image format validation (`jpg`, `jpeg`, `png`, `webp`, `bmp`, `tiff`)
- Max file size validation (`100MB`) for uploads
- Single-face policy messaging (rejects images with multiple faces in demo flow)
- Placeholder "no face detected" flow
- Placeholder prediction output:
  - `label` (`Real` or `AI-generated`)
  - `confidence` (`0-100%`)
  - `reasoning` list
- Threshold rule in demo flow: `AI-generated` when AI confidence is `>= 50%`
- Mobile-friendly UI

## Not included yet
- Real face detection
- Real AI-vs-real model inference
- Backend API

Current behavior is mocked in `app.js`.

## Local run
Because this is plain static HTML/CSS/JS, you can open `index.html` directly or run:

```bash
python3 -m http.server 8000
```

Then visit `http://localhost:8000`.

## Deploy to GitHub Pages
1. Create a GitHub repository.
2. Push this project.
3. In GitHub repo settings, enable Pages from the default branch root.
4. Your site will be available at `https://<your-username>.github.io/<repo-name>/`.

## Suggested next step
Replace `mockAnalyze()` in `app.js` with real backend/API calls once model inference is ready.
