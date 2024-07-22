document.querySelectorAll('.rangilalls-rvsf-landing-faqs-accordion').forEach(accordion => {
    accordion.addEventListener('click', function (e) {
        // Prevent click event from bubbling up
        e.stopPropagation();

        const panel = this.nextElementSibling;
        const isActive = this.classList.contains('active');

        // Check if the clicked accordion is a top-level or nested accordion
        const isNestedAccordion = this.closest('.rangilalls-rvsf-landing-faqs-panel') && this.closest('.rangilalls-rvsf-landing-faqs-panel').previousElementSibling.classList.contains('rangilalls-rvsf-landing-faqs-accordion');

        if (!isNestedAccordion) {
            // Close all top-level panels
            document.querySelectorAll('.rangilalls-rvsf-landing-faqs-accordion').forEach(acc => {
                if (!acc.closest('.rangilalls-rvsf-landing-faqs-panel')) {
                    acc.classList.remove('active');
                    if (acc.querySelector('.rangilalls-rvsf-landing-faqs-icon')) {
                        acc.querySelector('.rangilalls-rvsf-landing-faqs-icon').innerHTML = '&gt;';
                    }
                    if (acc.nextElementSibling) {
                        acc.nextElementSibling.classList.remove('show');
                    }
                }
            });
        } else {
            // Close all nested panels within the same top-level panel
            this.closest('.rangilalls-rvsf-landing-faqs-panel').querySelectorAll('.rangilalls-rvsf-landing-faqs-panel .rangilalls-rvsf-landing-faqs-accordion').forEach(acc => {
                acc.classList.remove('active');
                if (acc.querySelector('.rangilalls-rvsf-landing-faqs-icon')) {
                    acc.querySelector('.rangilalls-rvsf-landing-faqs-icon').innerHTML = '&gt;';
                }
                if (acc.nextElementSibling) {
                    acc.nextElementSibling.classList.remove('show');
                }
            });
        }

        // Open/close the clicked panel
        if (!isActive) {
            this.classList.add('active');
            if (panel) {
                panel.classList.add('show');
            }
            if (this.querySelector('.rangilalls-rvsf-landing-faqs-icon')) {
                this.querySelector('.rangilalls-rvsf-landing-faqs-icon').innerHTML = 'v';
            }
        } else {
            this.classList.remove('active');
            if (panel) {
                panel.classList.remove('show');
            }
            if (this.querySelector('.rangilalls-rvsf-landing-faqs-icon')) {
                this.querySelector('.rangilalls-rvsf-landing-faqs-icon').innerHTML = '&gt;';
            }
        }
    });
});
