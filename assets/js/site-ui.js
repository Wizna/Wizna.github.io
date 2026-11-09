(function($) {
    'use strict';

    $(function() {
        var $modal = $('#searchModal');
        var opener;
        var motion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)');
        // Bootstrap detects transition support once at load; keep it so the preference can be undone.
        var nativeTransition = $.support.transition;

        function updateMotion() {
            var reduced = !!(motion && motion.matches);
            // Bootstrap waits for transition events when a modal has .fade.
            $modal.toggleClass('fade', !reduced);
            // Collapse waits on a fallback timer for a transitionend that never comes once CSS
            // removes the transition, leaving it stuck in .collapsing and ignoring clicks.
            // With no transition support Bootstrap finishes immediately instead.
            $.support.transition = reduced ? false : nativeTransition;
        }

        updateMotion();
        if (motion && motion.addEventListener) {
            motion.addEventListener('change', updateMotion);
        }

        $modal.on('show.bs.modal', function() {
            opener = document.activeElement;
            var query = $('#tipue_search_input').val().trim();
            $modal.find('.search-modal__subtitle').text(query ? '与“' + query + '”相关的文章' : '按关键词查找文章');
            $modal.find('.search-modal__body').scrollTop(0);
        });
        $modal.on('shown.bs.modal', function() {
            $modal.find('.search-modal__close').trigger('focus');
        });
        $modal.on('hidden.bs.modal', function() {
            if (opener && document.documentElement.contains(opener)) {
                opener.focus({ preventScroll: true });
            }
        });

        // Keep wide tables inside the reading column on small screens.
        $('.entry table').each(function() {
            if ($(this).parent().hasClass('table-scroll')) {
                return;
            }
            $(this).wrap('<div class="table-scroll" tabindex="0" role="region" aria-label="表格，可横向滚动"></div>');
        });
    });
})(jQuery);
