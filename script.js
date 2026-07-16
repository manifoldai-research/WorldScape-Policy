const navToggle = document.querySelector(".nav-toggle");
const navLinks = document.querySelector(".nav-links");

if (navToggle && navLinks) {
  navToggle.addEventListener("click", () => {
    const isOpen = navLinks.classList.toggle("is-open");
    navToggle.setAttribute("aria-expanded", String(isOpen));
  });

  navLinks.addEventListener("click", (event) => {
    if (event.target instanceof HTMLAnchorElement) {
      navLinks.classList.remove("is-open");
      navToggle.setAttribute("aria-expanded", "false");
    }
  });
}

document.querySelectorAll("[data-copy-target]").forEach((button) => {
  button.addEventListener("click", async () => {
    const targetId = button.getAttribute("data-copy-target");
    const target = targetId ? document.getElementById(targetId) : null;
    if (!target) return;

    try {
      await navigator.clipboard.writeText(target.textContent.trim());
      const original = button.textContent;
      button.textContent = "Copied";
      window.setTimeout(() => {
        button.textContent = original;
      }, 1400);
    } catch {
      button.textContent = "Select";
    }
  });
});

const revealItems = document.querySelectorAll(
  ".section-heading, .contributions article, .figure-panel, .figure-card, .video-card, .showcase-item, .table-card, .training-flow",
);

if ("IntersectionObserver" in window) {
  const revealObserver = new IntersectionObserver(
    (entries, observer) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    },
    { threshold: 0.12, rootMargin: "0px 0px -40px" },
  );

  revealItems.forEach((item, index) => {
    item.classList.add("reveal");
    item.style.transitionDelay = `${Math.min(index % 4, 3) * 70}ms`;
    revealObserver.observe(item);
  });
}

document.querySelectorAll(".video-card video").forEach((video) => {
  video.addEventListener("mouseenter", () => video.play().catch(() => {}));
  video.addEventListener("mouseleave", () => video.pause());
});

document.querySelectorAll(".showcase-item video").forEach((video) => {
  video.play().catch(() => {});
});

const showcaseScroller = document.querySelector(".showcase-scroll");

if (showcaseScroller && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
  let isPaused = false;
  let lastTimestamp = null;
  let isResetting = false;

  showcaseScroller.addEventListener("mouseenter", () => {
    isPaused = true;
  });

  showcaseScroller.addEventListener("mouseleave", () => {
    isPaused = false;
  });

  const scrollShowcase = (timestamp) => {
    if (lastTimestamp === null) lastTimestamp = timestamp;
    const elapsed = timestamp - lastTimestamp;
    lastTimestamp = timestamp;

    if (!isPaused && !isResetting) {
      showcaseScroller.scrollLeft += elapsed * 0.06;
      const maxScrollLeft = showcaseScroller.scrollWidth - showcaseScroller.clientWidth;
      if (maxScrollLeft > 0 && showcaseScroller.scrollLeft >= maxScrollLeft - 2) {
        isResetting = true;
        showcaseScroller.style.scrollBehavior = "auto";
        showcaseScroller.scrollLeft = 0;
        window.setTimeout(() => {
          showcaseScroller.style.scrollBehavior = "";
          isResetting = false;
        }, 80);
      }
    }

    window.requestAnimationFrame(scrollShowcase);
  };

  window.requestAnimationFrame(scrollShowcase);
}
