(function() {
    'use strict';

    var root = document.documentElement;
    var storageKey = 'site-theme';
    var theme = 'dark';

    try {
        if (window.localStorage.getItem(storageKey) === 'light') {
            theme = 'light';
        }
    } catch (error) {
        // The default theme also works when browser storage is unavailable.
    }

    function applyTheme(nextTheme) {
        theme = nextTheme === 'light' ? 'light' : 'dark';
        root.setAttribute('data-theme', theme);
        var color = document.querySelector('meta[name="theme-color"]');
        if (color) {
            color.content = theme === 'dark' ? '#111722' : '#ffffff';
        }
        var button = document.querySelector('[data-theme-toggle]');
        if (button) {
            var label = theme === 'dark' ? '切换到浅色模式' : '切换到深色模式';
            button.setAttribute('aria-label', label);
            button.setAttribute('aria-pressed', theme === 'light' ? 'true' : 'false');
            button.title = label;
        }
    }

    // This script runs before stylesheets, so the first paint uses the saved theme.
    applyTheme(theme);

    function setTheme(nextTheme) {
        if (nextTheme === theme) {
            return;
        }
        applyTheme(nextTheme);
        document.dispatchEvent(new CustomEvent('site:themechange', { detail: { theme: theme } }));
    }

    function initToggle() {
        var button = document.querySelector('[data-theme-toggle]');
        if (!button) {
            return;
        }
        applyTheme(theme);
        button.hidden = false;
        button.addEventListener('click', function() {
            setTheme(theme === 'dark' ? 'light' : 'dark');
            try {
                window.localStorage.setItem(storageKey, theme);
            } catch (error) {
                // Switching remains available for this page without persistence.
            }
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initToggle);
    } else {
        initToggle();
    }

    window.addEventListener('storage', function(event) {
        if (event.key === storageKey || event.key === null) {
            setTheme(event.newValue === 'light' ? 'light' : 'dark');
        }
    });
})();
