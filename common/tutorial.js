/* HIP / CK-Tile 讲义公共脚本
 * 用法：在 <head> 中 <script src="../common/tutorial.js"></script>（不要加 defer，
 * 以便尽早应用主题、并让页面脚本可以随时调用 TUT.registerWidget）。
 * 导出：window.TUT = { registerWidget(name, fn) }
 * DOMContentLoaded 时：生成目录、滚动高亮当前节、代码高亮（hljs 可选）、
 *   渲染 assignment-card、调用已注册 widget 的 fn(el)。
 */
(function () {
  "use strict";

  var THEME_KEY = "hip-tut-theme";
  var root = document.documentElement;

  /* ---------- 主题 ---------- */
  function storedTheme() {
    try {
      var v = window.localStorage.getItem(THEME_KEY);
      return v === "dark" || v === "light" ? v : null;
    } catch (e) { return null; }
  }
  function saveTheme(v) {
    try { window.localStorage.setItem(THEME_KEY, v); } catch (e) { /* 隐私模式等：忽略 */ }
  }
  function systemDark() {
    return !!(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
  }
  function effectiveTheme() {
    var t = root.getAttribute("data-theme");
    if (t === "dark" || t === "light") return t;
    return systemDark() ? "dark" : "light";
  }
  var initial = storedTheme();
  if (initial) root.setAttribute("data-theme", initial);

  var ICON_SUN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4.5"/><path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8"/></svg>';
  var ICON_MOON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/></svg>';

  function buildThemeToggle() {
    if (document.querySelector(".theme-toggle")) return;
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "theme-toggle";
    function paint() {
      var dark = effectiveTheme() === "dark";
      btn.innerHTML = (dark ? ICON_SUN : ICON_MOON) + "<span>" + (dark ? "浅色" : "深色") + "</span>";
      btn.setAttribute("aria-label", dark ? "切换到浅色模式" : "切换到深色模式");
    }
    btn.addEventListener("click", function () {
      var next = effectiveTheme() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      saveTheme(next);
      paint();
    });
    if (window.matchMedia) {
      var mq = window.matchMedia("(prefers-color-scheme: dark)");
      var onChange = function () { paint(); };
      if (mq.addEventListener) mq.addEventListener("change", onChange);
      else if (mq.addListener) mq.addListener(onChange);
    }
    paint();
    document.body.appendChild(btn);
  }

  /* ---------- 目录 ---------- */
  function headingText(h) {
    // 去掉标题里的锚点符号等装饰，保留节号
    return (h.textContent || "").replace(/\s+/g, " ").trim();
  }

  function buildToc() {
    var nav = document.querySelector("nav.toc");
    var content = document.querySelector("main.content") || document.body;
    if (!nav) return null;
    var heads = Array.prototype.slice.call(content.querySelectorAll("h2[id], h3[id]"));
    if (!heads.length) return null;

    var toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "toc-toggle";
    toggle.setAttribute("aria-expanded", "false");
    toggle.innerHTML = '<span>目录</span><span class="cur"></span>';
    toggle.addEventListener("click", function () {
      var open = nav.classList.toggle("open");
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    });

    var list = document.createElement("ol");
    list.className = "toc-list";
    var links = [];
    heads.forEach(function (h) {
      var li = document.createElement("li");
      li.className = h.tagName === "H2" ? "lvl-2" : "lvl-3";
      var a = document.createElement("a");
      a.href = "#" + h.id;
      var sec = h.querySelector(".sec");
      if (sec) {
        var sp = document.createElement("span");
        sp.className = "sec";
        sp.textContent = sec.textContent.trim();
        a.appendChild(sp);
        var rest = h.cloneNode(true);
        rest.removeChild(rest.querySelector(".sec"));
        a.appendChild(document.createTextNode(headingText(rest)));
      } else {
        a.textContent = headingText(h);
      }
      a.addEventListener("click", function () {
        nav.classList.remove("open");
        toggle.setAttribute("aria-expanded", "false");
      });
      a.setAttribute("data-label", (sec ? sec.textContent.trim() + " " : "") + headingText(sec ? (function () { var c = h.cloneNode(true); c.removeChild(c.querySelector(".sec")); return c; })() : h));
      li.appendChild(a);
      list.appendChild(li);
      links.push(a);
    });
    nav.setAttribute("aria-label", nav.getAttribute("aria-label") || "本页目录");
    nav.appendChild(toggle);
    nav.appendChild(list);
    return { nav: nav, heads: heads, links: links, cur: toggle.querySelector(".cur") };
  }

  function trackActive(toc) {
    if (!toc) return;
    var activeIdx = -1;
    var ticking = false;
    function update() {
      ticking = false;
      var line = Math.min(window.innerHeight * 0.25, 160);
      var idx = 0;
      for (var i = 0; i < toc.heads.length; i++) {
        if (toc.heads[i].getBoundingClientRect().top - line <= 0) idx = i; else break;
      }
      // 滚到页底时高亮最后一节
      if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) idx = toc.heads.length - 1;
      if (idx === activeIdx) return;
      if (activeIdx >= 0) toc.links[activeIdx].classList.remove("active");
      activeIdx = idx;
      var a = toc.links[idx];
      a.classList.add("active");
      a.setAttribute("aria-current", "location");
      toc.links.forEach(function (l, j) { if (j !== idx) l.removeAttribute("aria-current"); });
      if (toc.cur) toc.cur.textContent = a.getAttribute("data-label") || a.textContent;
      // 让当前项在侧栏可见（仅侧栏可滚动时）
      var nav = toc.nav;
      if (nav.scrollHeight > nav.clientHeight + 4) {
        var top = a.offsetTop, h = a.offsetHeight;
        if (top < nav.scrollTop + 40 || top + h > nav.scrollTop + nav.clientHeight - 40) {
          nav.scrollTop = top - nav.clientHeight / 3;
        }
      }
    }
    function onScroll() { if (!ticking) { ticking = true; window.requestAnimationFrame(update); } }
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    update();
  }

  /* ---------- 代码块 ---------- */
  function langOf(code) {
    var m = /(?:^|\s)language-([\w+-]+)/.exec(code.className || "");
    return m ? m[1] : null;
  }

  // highlight.js 的 cpp 语法不认识 HIP 限定符，这里补上（只处理未被高亮的文本节点）
  var HIP_WORDS = /\b(__global__|__device__|__host__|__shared__|__launch_bounds__|__forceinline__|__restrict__|__syncthreads|__builtin_amdgcn_\w+)\b/g;
  function markHip(code) {
    var walker = document.createTreeWalker(code, NodeFilter.SHOW_TEXT, null);
    var nodes = [];
    while (walker.nextNode()) {
      var n = walker.currentNode;
      var par = n.parentNode;
      if (par !== code && par.className && /hljs-(comment|string)/.test(par.className)) continue;
      HIP_WORDS.lastIndex = 0;
      if (HIP_WORDS.test(n.nodeValue)) nodes.push(n);
    }
    nodes.forEach(function (n) {
      var frag = document.createDocumentFragment();
      var s = n.nodeValue, last = 0, m;
      HIP_WORDS.lastIndex = 0;
      while ((m = HIP_WORDS.exec(s))) {
        if (m.index > last) frag.appendChild(document.createTextNode(s.slice(last, m.index)));
        var sp = document.createElement("span");
        sp.className = /^__builtin|__syncthreads/.test(m[1]) ? "hljs-built_in" : "hljs-keyword";
        sp.textContent = m[1];
        frag.appendChild(sp);
        last = m.index + m[1].length;
      }
      if (last < s.length) frag.appendChild(document.createTextNode(s.slice(last)));
      n.parentNode.replaceChild(frag, n);
    });
  }

  function markTodos(code) {
    // 把注释里的 TODO(aK-n) 标亮；只处理文本节点，不破坏 hljs 结构
    var walker = document.createTreeWalker(code, NodeFilter.SHOW_TEXT, null);
    var nodes = [];
    while (walker.nextNode()) {
      if (/TODO\([\w-]+\)|>>> 你的代码 <<</.test(walker.currentNode.nodeValue)) nodes.push(walker.currentNode);
    }
    nodes.forEach(function (n) {
      var frag = document.createDocumentFragment();
      var re = /(TODO\([\w-]+\)|>>> 你的代码 <<<)/g;
      var s = n.nodeValue, last = 0, m;
      while ((m = re.exec(s))) {
        if (m.index > last) frag.appendChild(document.createTextNode(s.slice(last, m.index)));
        var sp = document.createElement("span");
        sp.className = "todo";
        sp.textContent = m[1];
        frag.appendChild(sp);
        last = m.index + m[1].length;
      }
      if (last < s.length) frag.appendChild(document.createTextNode(s.slice(last)));
      n.parentNode.replaceChild(frag, n);
    });
  }

  var LANG_LABEL = { cpp: "C++ / HIP", bash: "Shell", python: "Python", text: "文本", plaintext: "文本" };

  function setupCode() {
    var blocks = document.querySelectorAll("pre > code");
    var hl = window.hljs && typeof window.hljs.highlightElement === "function" ? window.hljs : null;
    if (hl && hl.configure) hl.configure({ ignoreUnescapedHTML: true });
    Array.prototype.forEach.call(blocks, function (code) {
      var pre = code.parentNode;
      var lang = langOf(code);
      if (lang && !pre.hasAttribute("data-lang")) pre.setAttribute("data-lang", LANG_LABEL[lang] || lang);
      if (hl && lang && hl.getLanguage && hl.getLanguage(lang)) {
        try { hl.highlightElement(code); } catch (e) { /* 高亮失败时保留纯文本 */ }
        if (lang === "cpp") markHip(code);
      }
      markTodos(code);
      addCopyButton(pre, function () { return code.textContent; });
    });
  }

  function copyText(text, btn) {
    function done(ok) {
      var old = btn.getAttribute("data-label") || btn.textContent;
      btn.setAttribute("data-label", old);
      btn.textContent = ok ? "已复制" : "复制失败";
      window.setTimeout(function () { btn.textContent = old; }, 1400);
    }
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(function () { done(true); }, function () { fallback(); });
    } else { fallback(); }
    function fallback() {
      try {
        var ta = document.createElement("textarea");
        ta.value = text;
        ta.setAttribute("readonly", "");
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        var ok = document.execCommand("copy");
        document.body.removeChild(ta);
        done(ok);
      } catch (e) { done(false); }
    }
  }

  function addCopyButton(container, getText) {
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn copy-btn";
    btn.textContent = "复制";
    btn.addEventListener("click", function () { copyText(getText(), btn); });
    container.appendChild(btn);
    return btn;
  }

  /* ---------- Assignment 工单 ---------- */
  // <div class="assignment-card" data-assign="a2" data-lab="lab1-hip-basics"
  //      data-path="lab1-hip-basics/a2-vector-add/vector_add.hip" data-title="向量加法">
  //   <p>任务简述（可选，任意 HTML）</p>
  // </div>
  function setupAssignments() {
    var cards = document.querySelectorAll(".assignment-card[data-assign]");
    Array.prototype.forEach.call(cards, function (card) {
      if (card.querySelector(".ac-body")) return; // 已渲染
      var id = card.getAttribute("data-assign");
      var title = card.getAttribute("data-title");
      var path = card.getAttribute("data-path");
      var lab = card.getAttribute("data-lab");
      var cmd = (lab ? "cd " + lab + " && " : "") + "make grade A=" + id;

      var desc = document.createElement("div");
      desc.className = "ac-desc";
      while (card.firstChild) desc.appendChild(card.firstChild);

      var stub = document.createElement("div");
      stub.className = "ac-stub";
      stub.innerHTML = '<span class="ac-label">Assignment</span><span class="ac-id"></span>';
      stub.querySelector(".ac-id").textContent = id;

      var body = document.createElement("div");
      body.className = "ac-body";
      var h = document.createElement("p");
      h.className = "ac-title";
      h.textContent = "现在去做 Assignment " + id + (title ? "：" + title : "");
      body.appendChild(h);
      if (desc.textContent.trim() || desc.children.length) body.appendChild(desc);

      var dl = document.createElement("dl");
      dl.className = "ac-rows";
      if (path) {
        var dt1 = document.createElement("dt"); dt1.textContent = "文件";
        var dd1 = document.createElement("dd");
        var p = document.createElement("span"); p.className = "ac-path"; p.textContent = path;
        dd1.appendChild(p);
        dl.appendChild(dt1); dl.appendChild(dd1);
      }
      var dt2 = document.createElement("dt"); dt2.textContent = "判分";
      var dd2 = document.createElement("dd");
      var box = document.createElement("div"); box.className = "ac-cmd";
      var c = document.createElement("code"); c.textContent = cmd;
      box.appendChild(c);
      dd2.appendChild(box);
      dl.appendChild(dt2); dl.appendChild(dd2);
      body.appendChild(dl);
      addCopyButton(box, function () { return cmd; });

      card.setAttribute("role", "note");
      card.setAttribute("aria-label", "Assignment " + id);
      card.appendChild(stub);
      card.appendChild(body);
    });
  }

  /* ---------- 交互小部件 ---------- */
  var registry = {};
  var ready = false;

  function mount(name, el) {
    if (el.__tutMounted) return;
    el.__tutMounted = true;
    try { registry[name](el); }
    catch (e) {
      console.error("[TUT] widget \"" + name + "\" 初始化失败：", e);
      var p = document.createElement("p");
      p.className = "widget-fallback";
      p.textContent = "这个交互小部件加载失败，不影响阅读正文。";
      el.appendChild(p);
    }
  }

  function mountAll(name) {
    var sel = name ? '.widget[data-widget="' + name + '"]' : ".widget[data-widget]";
    Array.prototype.forEach.call(document.querySelectorAll(sel), function (el) {
      var n = el.getAttribute("data-widget");
      if (registry[n]) mount(n, el);
    });
  }

  function registerWidget(name, fn) {
    if (typeof fn !== "function") throw new TypeError("TUT.registerWidget: fn 必须是函数");
    registry[name] = fn;
    if (ready) mountAll(name); // DOMContentLoaded 之后注册的也能挂载
  }

  /* ---------- 小工具（供 widget 使用） ---------- */
  var SVG_NS = "http://www.w3.org/2000/svg";
  function svg(tag, attrs, parent) {
    var n = document.createElementNS(SVG_NS, tag);
    if (attrs) for (var k in attrs) if (Object.prototype.hasOwnProperty.call(attrs, k)) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }

  function init() {
    if (ready) return;
    buildThemeToggle();
    trackActive(buildToc());
    setupCode();
    setupAssignments();
    ready = true;
    mountAll();
  }

  window.TUT = { registerWidget: registerWidget, svg: svg, copyText: copyText };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
