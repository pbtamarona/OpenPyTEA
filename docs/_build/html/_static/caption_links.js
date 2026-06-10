document.addEventListener("DOMContentLoaded", function () {
    // Derive root path from a known root-level sidebar link.
    // quickstart/installation are always in the "Getting Started" toctree at
    // the docs root, so their href is either "quickstart.html" (root page) or
    // "../quickstart.html" (subdirectory page) — stripping the filename gives
    // the correct relative prefix regardless of where we are.
    var anchor = document.querySelector(
        'a.reference.internal[href$="quickstart.html"], ' +
        'a.reference.internal[href$="installation.html"]'
    );
    var root = anchor ? anchor.getAttribute("href").replace(/[^/]+\.html$/, "") : "";

    var captionLinks = {
        "User Guide": root + "user_guide/index.html",
        "API Reference": root + "api/index.html"
    };

    document.querySelectorAll(".sidebar-tree .caption-text").forEach(function (el) {
        var text = el.textContent.trim();
        if (captionLinks[text]) {
            var a = document.createElement("a");
            a.href = captionLinks[text];
            a.textContent = text;
            a.className = "caption-link";
            el.textContent = "";
            el.appendChild(a);
        }
    });
});
