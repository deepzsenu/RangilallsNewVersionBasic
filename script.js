let slideIndex = 0;
showSlides();

function showSlides() {
  let slides = document.getElementsByClassName("rangilalls-header-slides");
  let dots = document.getElementsByClassName("rangilalls-header-dot");
  for (let i = 0; i < slides.length; i++) {
    slides[i].style.display = "none";
  }
  slideIndex++;
  if (slideIndex > slides.length) {
    slideIndex = 1;
  }
  for (let i = 0; i < dots.length; i++) {
    dots[i].className = dots[i].className.replace(" active", "");
  }
  slides[slideIndex - 1].style.display = "block";
  dots[slideIndex - 1].className += " active";
  setTimeout(showSlides, 2000); // Change image every 2 seconds
}

function currentSlide(n) {
  slideIndex = n;
  showSlides();
}

/**
 * Adds a click event listener to an accordion element to toggle its active state and display its associated panel.
 *
 * @param {Element} accordion - The accordion element to which the click event listener is added.
 *
 * The function toggles the active state of the clicked accordion element and its associated panel.
 * It also ensures that only one top-level or nested panel is open at a time.
 *
 * @returns {void} This function does not return a value.
 */
document
  .querySelectorAll(".rangilalls-rvsf-landing-faqs-accordion")
  .forEach((accordion) => {
    accordion.addEventListener("click", function (e) {
      // Prevent click event from bubbling up
      e.stopPropagation();

      const panel = this.nextElementSibling;
      const isActive = this.classList.contains("active");

      // Check if the clicked accordion is a top-level or nested accordion
      const isNestedAccordion =
        this.closest(".rangilalls-rvsf-landing-faqs-panel") &&
        this.closest(
          ".rangilalls-rvsf-landing-faqs-panel"
        ).previousElementSibling.classList.contains(
          "rangilalls-rvsf-landing-faqs-accordion"
        );

      if (!isNestedAccordion) {
        // Close all top-level panels
        document
          .querySelectorAll(".rangilalls-rvsf-landing-faqs-accordion")
          .forEach((acc) => {
            if (!acc.closest(".rangilalls-rvsf-landing-faqs-panel")) {
              acc.classList.remove("active");
              if (acc.querySelector(".rangilalls-rvsf-landing-faqs-icon")) {
                acc.querySelector(
                  ".rangilalls-rvsf-landing-faqs-icon"
                ).innerHTML = "&gt;";
              }
              if (acc.nextElementSibling) {
                acc.nextElementSibling.classList.remove("show");
              }
            }
          });
      } else {
        // Close all nested panels within the same top-level panel
        this.closest(".rangilalls-rvsf-landing-faqs-panel")
          .querySelectorAll(
            ".rangilalls-rvsf-landing-faqs-panel .rangilalls-rvsf-landing-faqs-accordion"
          )
          .forEach((acc) => {
            acc.classList.remove("active");
            if (acc.querySelector(".rangilalls-rvsf-landing-faqs-icon")) {
              acc.querySelector(
                ".rangilalls-rvsf-landing-faqs-icon"
              ).innerHTML = "&gt;";
            }
            if (acc.nextElementSibling) {
              acc.nextElementSibling.classList.remove("show");
            }
          });
      }

      // Open/close the clicked panel
      if (!isActive) {
        this.classList.add("active");
        if (panel) {
          panel.classList.add("show");
        }
        if (this.querySelector(".rangilalls-rvsf-landing-faqs-icon")) {
          this.querySelector(".rangilalls-rvsf-landing-faqs-icon").innerHTML =
            "v";
        }
      } else {
        this.classList.remove("active");
        if (panel) {
          panel.classList.remove("show");
        }
        if (this.querySelector(".rangilalls-rvsf-landing-faqs-icon")) {
          this.querySelector(".rangilalls-rvsf-landing-faqs-icon").innerHTML =
            "&gt;";
        }
      }
    });
  });

// server time fetxhing
async function fetchServerTime() {
    try {
      const response = await fetch('http://worldtimeapi.org/api/timezone/Asia/Kolkata');
      if (!response.ok) {
        throw new Error('Network response was not ok ' + response.statusText);
      }
      const serverTime = await response.json();
      document.querySelector('.rangilalls-elv-footer-servertime').textContent = `Server Clock: ${new Date(serverTime.datetime).toLocaleString()}`;
    } catch (error) {
      console.error('There has been a problem with your fetch operation:', error);
    }
  }
  
  // Call the function to fetch and display the server time
  fetchServerTime();
  
  // Optionally, you can set an interval to update the time periodically
  setInterval(fetchServerTime, 600); // Update every 60 seconds