/**
 * 全局顶栏公告（AnnouncementBar）
 * 可配置 id / message / cta / href / variant（glass | soft），关闭后按 id 存 localStorage 不再展示。
 */
(function () {
    var LS_PREFIX = 'announce-dismissed-';

    function getConfig() {
        var root = document.getElementById('announcement-bar-root');
        if (!root) return null;
        var raw = root.getAttribute('data-config');
        if (!raw) return null;
        try {
            return JSON.parse(raw);
        } catch (e) {
            return null;
        }
    }

    function isDismissed(id) {
        try {
            return localStorage.getItem(LS_PREFIX + id) === '1';
        } catch (e) {
            return false;
        }
    }

    function setDismissed(id) {
        try {
            localStorage.setItem(LS_PREFIX + id, '1');
        } catch (e) {}
    }

    function render(config) {
        var id = config.id || 'announce-default';
        var message = config.message || '';
        var cta = config.cta || '';
        var href = config.href || '#';

        var root = document.getElementById('announcement-bar-root');
        root.setAttribute('role', 'banner');

        var inner = document.createElement('div');
        inner.className = 'announcement-bar__inner';

        var badge = document.createElement('span');
        badge.className = 'announcement-bar__badge';
        badge.textContent = '公告';

        var msg = document.createElement('p');
        msg.className = 'announcement-bar__message';
        msg.textContent = message;

        var right = document.createElement('div');
        right.className = 'announcement-bar__right';
        if (cta) {
            var a = document.createElement('a');
            a.className = 'announcement-bar__cta';
            a.href = href;
            a.textContent = cta;
            right.appendChild(a);
        }
        var closeBtn = document.createElement('button');
        closeBtn.type = 'button';
        closeBtn.className = 'announcement-bar__close';
        closeBtn.setAttribute('aria-label', '关闭公告');
        closeBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M18.3 5.7a1 1 0 0 0-1.4 0L12 10.6 7.1 5.7A1 1 0 0 0 5.7 7.1l4.9 4.9-4.9 4.9a1 1 0 1 0 1.4 1.4l4.9-4.9 4.9 4.9a1 1 0 0 0 1.4-1.4l-4.9-4.9 4.9-4.9a1 1 0 0 0 0-1.4z"/></svg>';
        closeBtn.addEventListener('click', function () {
            setDismissed(id);
            root.style.display = 'none';
        });
        right.appendChild(closeBtn);

        inner.appendChild(badge);
        inner.appendChild(msg);
        inner.appendChild(right);
        root.innerHTML = '';
        root.appendChild(inner);
    }

    function init() {
        var config = getConfig();
        if (!config || !config.id) return;
        if (isDismissed(config.id)) {
            var root = document.getElementById('announcement-bar-root');
            if (root) root.style.display = 'none';
            return;
        }
        render(config);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
