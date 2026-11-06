(function() {
    'use strict';

    function initDiagrams() {
        var nodes = Array.from(document.querySelectorAll('.language-mermaid, .mermaid'));
        if (!nodes.length) {
            return;
        }
        var sources = nodes.map(function(node) { return node.textContent; });
        var queue = Promise.resolve();

        function render() {
            // Restore source text and serialize renders when the theme changes quickly.
            queue = queue.then(function() {
                window.mermaid.initialize({
                    startOnLoad: false,
                    theme: document.documentElement.getAttribute('data-theme') === 'light' ? 'default' : 'dark'
                });
                nodes.forEach(function(node, index) {
                    node.textContent = sources[index];
                    node.removeAttribute('data-processed');
                });
                return window.mermaid.run({ nodes: nodes });
            }).catch(function(error) {
                console.error('Unable to render Mermaid diagrams', error);
            });
        }

        var script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js';
        script.async = true;
        script.onload = function() {
            render();
            document.addEventListener('site:themechange', render);
        };
        document.head.appendChild(script);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initDiagrams);
    } else {
        initDiagrams();
    }
})();
