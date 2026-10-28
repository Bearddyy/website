# website
Personal Website

## Photography uploads

Photos are stored on the server under `data/photos/` (git-ignored), so they
never live in the repo. Uploads are authenticated with the `PHOTO_UPLOAD_KEY`
environment variable: set it in the server environment only, never in source
or committed config.

    PHOTO_UPLOAD_KEY=<a long random secret> gunicorn website:app

Upload a photo (run wherever the secret is kept):

    curl -F photo=@shot.jpg -H "X-Upload-Key: $PHOTO_UPLOAD_KEY" \
        https://your-domain/photos/upload

Accepted formats: JPEG, PNG, WEBP, TIFF, BMP, MPO (camera JPEGs). Each upload
is re-encoded to a full resolution JPEG plus a 1600px display copy for the
grid. Without the header/key: 401; with `PHOTO_UPLOAD_KEY` unset the endpoint
is disabled (503).
Photos newer than that render at the top of `/photography`, above a `\ SEEN`
divider. The SEEN button marks everything seen, CLEAR wipes the cookie so
everything counts as new again.

Note: Render's filesystem is ephemeral unless a persistent disk is attached,
so mount one covering `data/photos/` or uploads vanish on redeploy.
