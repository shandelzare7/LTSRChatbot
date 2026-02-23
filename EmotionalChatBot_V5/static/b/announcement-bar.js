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
        var variant = (config.variant === 'soft') ? 'soft' : 'glass';

        var bar = document.createElement('div');
        bar.className = 'announcement-bar announcement-bar--' + variant;
        bar.setAttribute('role', 'banner');

        var inner = document.createElement('div');
        inner.className = 'announcement-bar__inner';

        var left = document.createElement('div');
        left.className = 'announcement-bar__left';
        var icon = document.createElement('span');
        icon.className = 'announcement-bar__icon';
        icon.setAttribute('aria-hidden', 'true');
        icon.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>';
        var msg = document.createElement('span');
        msg.className = 'announcement-bar__message';
        msg.textContent = message;
        left.appendChild(icon);
        left.appendChild(msg);

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
        closeBtn.textContent = '×';
        closeBtn.addEventListener('click', function () {
            setDismissed(id);
            root.style.display = 'none';
        });
        right.appendChild(closeBtn);

        inner.appendChild(left);
        inner.appendChild(right);
        bar.appendChild(inner);

        var root = document.getElementById('announcement-bar-root');
        root.innerHTML = '';
        root.appendChild(bar);
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
