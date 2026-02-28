// Service Worker for notification support.
// Note: This is NOT Web Push (no remote push when browser is closed).
// It enables showing notifications reliably while the site is open/backgrounded.
//
// Scope: served at /sw.js so it can control '/'.

self.addEventListener('install', (event) => {
  // Activate immediately after install.
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener('push', (event) => {
  event.waitUntil((async () => {
    let payload = {};
    try {
      payload = event.data ? event.data.json() : {};
    } catch (e) {
      try {
        payload = { body: event.data ? event.data.text() : '' };
      } catch (e2) {
        payload = {};
      }
    }
    const title = payload.title || 'Chatbot';
    const body = payload.body || '';
    const url = payload.url || '/';

    // If user is actively looking at the page, skip the notification
    try {
      const allClients = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
      const hasVisible = allClients.some((c) => {
        try { return c.visibilityState === 'visible'; } catch (e) { return false; }
      });
      if (hasVisible) return;
    } catch (e) {}

    await self.registration.showNotification(title, {
      body,
      tag: payload.tag || 'ltsr-push',
      data: { url },
      renotify: true,
    });
  })());
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  event.waitUntil((async () => {
    const url = (event.notification && event.notification.data && event.notification.data.url) ? event.notification.data.url : '/';
    const allClients = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
    for (const c of allClients) {
      if ('focus' in c) {
        await c.focus();
        try {
          if (url && c.navigate) await c.navigate(url);
        } catch (e) {}
        return;
      }
    }
    if (self.clients.openWindow) {
      await self.clients.openWindow(url || '/');
    }
  })());
});

