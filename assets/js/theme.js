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

    var switching = 0;

    function nextFrame(callback) {
        if (window.requestAnimationFrame) {
            window.requestAnimationFrame(callback);
        } else {
            window.setTimeout(callback, 16);
        }
    }

    // Per-element colour transitions run out of step with the page background, so they are
    // switched off (see .theme-switching in _site.scss) until the new theme has been painted.
    function holdTransitions() {
        switching += 1;
        root.classList.add('theme-switching');
    }

    function releaseTransitions() {
        switching = Math.max(0, switching - 1);
        if (!switching) {
            root.classList.remove('theme-switching');
        }
    }

    function setTheme(nextTheme) {
        if (nextTheme === theme) {
            return;
        }
        // The saved state changes now; the page itself changes when commit() runs.
        theme = nextTheme;

        function commit() {
            applyTheme(theme);
            document.dispatchEvent(new CustomEvent('site:themechange', { detail: { theme: theme } }));
        }

        holdTransitions();
        var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        if (document.startViewTransition && !reduceMotion) {
            // Cross-fade the whole page so every colour changes together.
            document.startViewTransition(commit).finished.then(releaseTransitions, releaseTransitions);
        } else {
            commit();
            // The first frame paints the new theme while transitions are still off.
            nextFrame(function() { nextFrame(releaseTransitions); });
        }
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
