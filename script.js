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

const activateVideo = (video) => {
  if (!(video instanceof HTMLVideoElement) || video.dataset.activated === "true") return;

  video.dataset.activated = "true";
  video.preload = "auto";

  if (video.hasAttribute("data-autoplay")) {
    video.muted = true;
    video.defaultMuted = true;
    video.autoplay = true;
    video.setAttribute("muted", "");
    video.setAttribute("autoplay", "");

    const startPlayback = () => video.play().catch(() => {});
    video.addEventListener("canplay", startPlayback, { once: true });
    video.load();

    if (video.readyState >= HTMLMediaElement.HAVE_FUTURE_DATA) {
      startPlayback();
    }
  } else {
    video.load();
  }
};

const lazyVideos = document.querySelectorAll("[data-lazy-video]");

if ("IntersectionObserver" in window) {
  const lazyVideoObserver = new IntersectionObserver(
    (entries, observer) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting || !(entry.target instanceof HTMLVideoElement)) return;
        activateVideo(entry.target);
        observer.unobserve(entry.target);
      });
    },
    { rootMargin: "50px 0px", threshold: 0.01 },
  );

  lazyVideos.forEach((video) => lazyVideoObserver.observe(video));
} else {
  lazyVideos.forEach(activateVideo);
}

document.querySelectorAll(".video-card video:not([data-prompt-video])").forEach((video) => {
  video.addEventListener("mouseenter", () => {
    activateVideo(video);
    video.play().catch(() => {});
  });
  video.addEventListener("mouseleave", () => video.pause());
});

const promptVideo = document.querySelector("[data-prompt-video]");

if (promptVideo instanceof HTMLVideoElement) {
  const carousel = promptVideo.nextElementSibling;
  const track = carousel?.querySelector(".prompt-track");
  const cards = Array.from(carousel?.querySelectorAll(".prompt-card") ?? []);
  const timelineItems = Array.from(carousel?.querySelectorAll(".prompt-timeline span") ?? []);
  const starts = cards.map((card) => Number(card.dataset.start ?? 0));
  let activeIndex = -1;
  let animationFrame = 0;

  const updatePrompt = () => {
    const currentTime = promptVideo.currentTime;
    const nextIndex = starts.findLastIndex((start) => currentTime >= start);
    const index = Math.max(0, nextIndex);

    if (index !== activeIndex) {
      activeIndex = index;
      cards.forEach((card, cardIndex) => card.classList.toggle("is-active", cardIndex === index));

      const activeCard = cards[index];
      if (track && activeCard instanceof HTMLElement) {
        const targetLeft = activeCard.offsetLeft - (track.clientWidth - activeCard.offsetWidth) / 2;
        track.scrollTo({ left: Math.max(0, targetLeft), behavior: "smooth" });
      }
    }

    timelineItems.forEach((item, itemIndex) => {
      const segmentStart = starts[itemIndex];
      const segmentEnd = starts[itemIndex + 1] ?? promptVideo.duration;
      const progress =
        itemIndex < index
          ? 100
          : itemIndex > index
            ? 0
            : Math.min(100, Math.max(0, ((currentTime - segmentStart) / (segmentEnd - segmentStart)) * 100));

      item.classList.toggle("is-active", progress > 0);
      item.style.background = `linear-gradient(to right, var(--purple) ${progress}%, #ddd3e8 ${progress}%)`;
    });
  };

  const animatePrompt = () => {
    updatePrompt();
    if (!promptVideo.paused) animationFrame = window.requestAnimationFrame(animatePrompt);
  };

  cards.forEach((card, index) => {
    card.addEventListener("click", () => {
      activateVideo(promptVideo);
      const seekAndPlay = () => {
        promptVideo.currentTime = starts[index];
        promptVideo.play().catch(() => {});
        updatePrompt();
      };

      if (promptVideo.readyState >= HTMLMediaElement.HAVE_METADATA) {
        seekAndPlay();
      } else {
        promptVideo.addEventListener("loadedmetadata", seekAndPlay, { once: true });
      }
    });
  });

  promptVideo.addEventListener("play", () => {
    window.cancelAnimationFrame(animationFrame);
    animationFrame = window.requestAnimationFrame(animatePrompt);
  });
  promptVideo.addEventListener("pause", () => window.cancelAnimationFrame(animationFrame));
  promptVideo.addEventListener("loadedmetadata", updatePrompt);
}

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

const currentDate = document.querySelector("#current-date");

if (currentDate instanceof HTMLTimeElement) {
  const updateCurrentDate = () => {
    const now = new Date();
    currentDate.dateTime = [
      now.getFullYear(),
      String(now.getMonth() + 1).padStart(2, "0"),
      String(now.getDate()).padStart(2, "0"),
    ].join("-");
    currentDate.textContent = new Intl.DateTimeFormat("en-US", {
      month: "long",
      day: "numeric",
      year: "numeric",
    }).format(now);
  };

  updateCurrentDate();
  window.setInterval(updateCurrentDate, 60_000);
}
