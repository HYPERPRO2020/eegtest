// POST /api/blob-upload-token — the one piece of this app that has to be
// Node, not Python: Vercel Blob's client-upload flow (browser -> Blob
// directly, bypassing the 4.5MB function-body limit, see ARCHITECTURE.md)
// is a token handshake coupled to the official @vercel/blob/client SDK,
// which only ships for JS. Everything else in this app is Python; this
// file exists solely to generate that token using the real SDK instead of
// hand-rolling its protocol from memory with no way to verify it against a
// live account in this environment (same "flag what's unverified" honesty
// as the rest of this repo -- this file specifically follows Vercel's own
// documented client-upload recipe as closely as possible).
//
// The browser calls `upload(filename, file, { access: 'public',
// handleUploadUrl: '/api/blob-upload-token' })` from `@vercel/blob/client`
// once per uploaded recording; this handler's only job is authorizing that
// upload and handing back a short-lived client token.

const { handleUpload } = require('@vercel/blob/client');

module.exports = async (request, response) => {
  const body = request.body;

  try {
    const jsonResponse = await handleUpload({
      body,
      request,
      onBeforeGenerateToken: async (pathname) => {
        return {
          allowedContentTypes: ['*/*'],
          addRandomSuffix: true, // avoid collisions between concurrent uploads
          tokenPayload: JSON.stringify({ pathname }),
        };
      },
      onUploadCompleted: async () => {
        // No server-side action needed on completion -- the browser already
        // holds every uploaded file's URL locally and calls /api/create_job
        // once the whole batch has finished uploading.
      },
    });

    response.status(200).json(jsonResponse);
  } catch (error) {
    response.status(400).json({ error: error.message });
  }
};
