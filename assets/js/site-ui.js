(function($) {
    'use strict';

    $(function() {
        var $modal = $('#searchModal');
        var opener;
        var motion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)');

        function updateMotion() {
            var reduced = !!(motion && motion.matches);
            // Bootstrap waits for transition events when a modal has .fade.
            $modal.toggleClass('fade', !reduced);
            if (reduced) {
                // Finish active collapses whose CSS transition has just been removed.
                // Keep $.support.transition intact: Bootstrap's pending timers still use it.
                $('.collapsing').trigger('bsTransitionEnd');
            }
        }

        $(document).on('show.bs.collapse hide.bs.collapse', function(event) {
            if (motion && motion.matches) {
                var panel = event.target;
                // Bootstrap attaches its completion handler after the show/hide event returns.
                window.setTimeout(function() {
                    if (motion.matches) {
                        $(panel).filter('.collapsing').trigger('bsTransitionEnd');
                    }
                }, 0);
            }
        });

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
