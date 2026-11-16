(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const elements = {
    body: document.body,
    sidebar: $("#sidebar"),
    messages: $("#messages"),
    chatScroll: $("#chat-scroll"),
    welcome: $("#welcome"),
    chatTitle: $("#chat-title"),
    input: $("#message-input"),
    form: $("#composer-form"),
    send: $("#send-button"),
    attach: $("#file-input"),
    pills: $("#upload-pills"),
    history: $("#conversation-list"),
    model: $("#model-select"),
    persona: $("#persona-select"),
    responseStyle: $("#response-style"),
    webSearch: $("#web-search-toggle"),
    voiceSelect: $("#voice-select"),
    speechRate: $("#speech-rate"),
    speechRateValue: $("#speech-rate-value"),
    engineBadge: $("#engine-badge"),
    engineLabel: $("#engine-label"),
    providerHint: $("#provider-hint"),
    welcomeStatus: $("#welcome-status"),
    toast: $("#toast-region"),
    themeToggle: $("#theme-toggle"),
    themeIcon: $("#theme-icon"),
    themeLabel: $("#theme-label"),
    voiceToggle: $("#voice-toggle"),
    dictate: $("#dictate-button"),
    install: $("#install-button"),
    renameDialog: $("#rename-dialog"),
    renameInput: $("#rename-input"),
    renameConfirm: $("#rename-confirm"),
  };

  const state = {
    conversationId: newId(),
    title: "New conversation",
    model: "",
    provider: "local",
    online: false,
    executionEnabled: false,
    visionModel: "qwen2.5vl:7b",
    busy: false,
    attachments: [],
    voiceEnabled: localStorage.getItem("voiceEnabled") === "true",
    recognition: null,
    deferredInstall: null,
    renameId: null,
    toastTimer: null,
    statusTimer: null,
  };

  function newId() {
    if (crypto.randomUUID) return crypto.randomUUID().replaceAll("-", "");
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    return [...bytes].map((value) => value.toString(16).padStart(2, "0")).join("");
  }

  function selectDocumentChunks(documents, query) {
    const terms = String(query || "").toLowerCase().match(/[\p{L}\p{N}]{3,}/gu) || [];
    const candidates = [];
    documents.forEach((file) => {
      (file.chunks || []).forEach((text, index) => {
        const lower = text.toLowerCase();
        const score = terms.reduce((total, term) => {
          let found = 0;
          let offset = 0;
          while ((offset = lower.indexOf(term, offset)) !== -1) {
            found += 1;
            offset += term.length;
          }
          return total + Math.min(found, 5);
        }, 0);
        candidates.push({ file: file.name, index, text, score });
      });
    });
    candidates.sort((left, right) => right.score - left.score || left.index - right.index);
    let remaining = 20_000;
    const selected = [];
    candidates.slice(0, 8).forEach((chunk) => {
      if (remaining <= 0) return;
      const content = chunk.text.slice(0, remaining);
      selected.push(`Document: ${chunk.file} (chunk ${chunk.index + 1})\n<document>\n${content}\n</document>`);
      remaining -= content.length;
    });
    return selected.join("\n\n");
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[char]);
  }

  function showToast(message, isError = false, duration = 3600) {
    elements.toast.textContent = message;
    elements.toast.classList.toggle("error", isError);
    elements.toast.classList.add("show");
    clearTimeout(state.toastTimer);
    state.toastTimer = setTimeout(() => elements.toast.classList.remove("show"), duration);
  }

  async function api(url, options = {}) {
    const response = await fetch(url, {
      credentials: "same-origin",
      ...options,
      headers: {
        ...(options.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
        ...options.headers,
      },
    });
    const contentType = response.headers.get("content-type") || "";
    if (!response.ok) {
      let detail = `Request failed (${response.status})`;
      if (contentType.includes("application/json")) {
        const body = await response.json().catch(() => ({}));
        detail = body.error || detail;
      } else {
        detail = (await response.text()).slice(0, 250) || detail;
      }
      throw new Error(detail);
    }
    return { response, contentType };
  }

  function scrollToLatest(smooth = true) {
    elements.chatScroll.scrollTo({
      top: elements.chatScroll.scrollHeight,
      behavior: smooth ? "smooth" : "instant",
    });
  }

  function setBusy(busy) {
    state.busy = busy;
    elements.send.disabled = busy;
    elements.attach.disabled = busy;
    elements.send.innerHTML = busy ? '<span class="send-spinner"></span>' : "<span>↑</span>";
    elements.send.setAttribute("aria-label", busy ? "AI scrie…" : "Send message");
  }

  function hideWelcome() {
    elements.welcome.hidden = true;
  }

  function setTitle(title) {
    state.title = title || "New conversation";
    elements.chatTitle.textContent = state.title;
    document.title = `${state.title} · All-in-One AI Studio`;
  }

  function createMessage(role, text, options = {}) {
    hideWelcome();
    const isUser = role === "user";
    const message = document.createElement("article");
    message.className = `message ${isUser ? "user" : "assistant"}`;
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = isUser ? "Y" : "✦";
    avatar.setAttribute("aria-hidden", "true");

    const content = document.createElement("div");
    content.className = "message-content";
    const meta = document.createElement("div");
    meta.className = "message-meta";
    const name = document.createElement("strong");
    name.textContent = isUser ? "You" : "AI Studio";
    const time = document.createElement("time");
    const now = new Date();
    time.dateTime = now.toISOString();
    time.textContent = now.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    meta.append(name, time);

    const card = document.createElement("div");
    card.className = "message-card";
    if (options.thinking) {
      card.innerHTML = '<span class="thinking">AI scrie…<span class="thinking-dots"><i></i><i></i><i></i></span></span>';
    } else if (isUser) {
      card.textContent = text;
      (options.images || []).forEach((image) => {
        const preview = document.createElement("img");
        preview.className = "message-image-preview";
        preview.src = `data:${image.mime_type};base64,${image.data}`;
        preview.alt = image.name || "Attached image";
        card.append(preview);
      });
    } else {
      renderMarkdown(card, text);
    }

    content.append(meta, card);
    if (!isUser && !options.thinking) {
      addMessageTools(content, text);
      if (options.sources?.length) addCitations(content, options.sources);
    }
    message.append(avatar, content);
    elements.messages.append(message);
    scrollToLatest();
    return { message, content, card };
  }

  function addMessageTools(container, content) {
    const tools = document.createElement("div");
    tools.className = "message-tools";
    const copy = document.createElement("button");
    copy.className = "message-tool";
    copy.type = "button";
    copy.textContent = "▣  Copy";
    copy.setAttribute("aria-label", "Copy assistant message");
    copy.addEventListener("click", () => copyText(content, "Message copied"));
    const speak = document.createElement("button");
    speak.className = "message-tool";
    speak.type = "button";
    speak.textContent = "◖  Read aloud";
    speak.setAttribute("aria-label", "Read assistant message aloud");
    speak.addEventListener("click", () => speakText(content));
    tools.append(copy, speak);
    container.append(tools);
  }

  function addCitations(container, sources) {
    const list = document.createElement("div");
    list.className = "citation-list";
    list.setAttribute("aria-label", "Web search sources");
    sources.forEach((source, index) => {
      if (!source || typeof source.url !== "string") return;
      try {
        const url = new URL(source.url);
        if (!["http:", "https:"].includes(url.protocol)) return;
        const link = document.createElement("a");
        link.className = "citation-badge";
        link.href = url.href;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = `${index + 1}. ${source.title || url.hostname}`;
        link.title = source.snippet || source.title || "";
        list.append(link);
      } catch (_error) {
        return;
      }
    });
    if (list.childElementCount) container.append(list);
  }

  function renderMath(target) {
    if (!window.renderMathInElement) return;
    renderMathInElement(target, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code", "option"],
    });
  }

  function renderMarkdown(target, markdown) {
    if (!window.marked || !window.DOMPurify) {
      target.textContent = markdown;
      renderMath(target);
      return;
    }
    const mathExpressions = [];
    const protectedMarkdown = markdown.replace(
      /\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\]|(^|[^$])(\$(?!\$)(?:\\.|[^\\$\n])+\$(?!\$))/g,
      (expression, prefix, inline) => {
        const math = inline || expression;
        const placeholder = `@@MATH_${mathExpressions.length}@@`;
        mathExpressions.push(math);
        return inline ? `${prefix}${placeholder}` : placeholder;
      },
    );
    const renderer = new marked.Renderer();
    renderer.link = ({ href, title, text }) => {
      let safeUrl = "#";
      try {
        const candidate = new URL(href, window.location.origin);
        if (["http:", "https:", "mailto:"].includes(candidate.protocol)) {
          safeUrl = candidate.href;
        }
      } catch (_error) {
        safeUrl = "#";
      }
      const titleAttr = title ? ` title="${escapeHtml(title)}"` : "";
      return `<a href="${escapeHtml(safeUrl)}" rel="noopener noreferrer" target="_blank"${titleAttr}>${text}</a>`;
    };
    const html = marked.parse(protectedMarkdown, { breaks: true, gfm: true, renderer });
    target.innerHTML = DOMPurify.sanitize(html, {
      USE_PROFILES: { html: true },
      ADD_ATTR: ["target", "rel"],
    });
    decorateCodeBlocks(target);
    if (mathExpressions.length) {
      const walker = document.createTreeWalker(target, NodeFilter.SHOW_TEXT);
      const placeholderPattern = /@@MATH_(\d+)@@/g;
      let textNode;
      while ((textNode = walker.nextNode())) {
        textNode.nodeValue = textNode.nodeValue.replace(
          placeholderPattern,
          (placeholder, index) => mathExpressions[Number(index)] || placeholder,
        );
      }
    }
    renderMath(target);
    $$("a", target).forEach((link) => {
      link.addEventListener("click", (event) => {
        const url = new URL(link.href, window.location.href);
        if (!["http:", "https:", "mailto:"].includes(url.protocol)) event.preventDefault();
      });
    });
  }

  function decorateCodeBlocks(root) {
    $$("pre", root).forEach((pre) => {
      const code = $("code", pre);
      if (!code || pre.closest(".code-block")) return;
      const language = [...code.classList].find((item) => item.startsWith("language-"));
      if (window.hljs && language !== "language-plotly") {
        try { hljs.highlightElement(code); } catch (_error) { /* Plain code remains readable. */ }
      }
      const wrapper = document.createElement("div");
      wrapper.className = "code-block";
      const toolbar = document.createElement("div");
      toolbar.className = "code-toolbar";
      const label = document.createElement("span");
      label.textContent = language ? language.slice("language-".length).toUpperCase() : "CODE";
      const copy = document.createElement("button");
      copy.className = "copy-code";
      copy.type = "button";
      copy.textContent = "Copy code";
      copy.addEventListener("click", () => copyText(code.textContent, "Code copied"));
      toolbar.append(label);
      if (language === "language-python" || language === "language-py") {
        const run = document.createElement("button");
        run.className = "run-code";
        run.type = "button";
        run.textContent = "▶ Run Code";
        run.disabled = !state.executionEnabled;
        run.title = state.executionEnabled
          ? "Run in the restricted local Python interpreter"
          : "Disabled in public/cloud mode";
        run.addEventListener("click", () => runCode(code.textContent, wrapper, run));
        toolbar.append(run);
      }
      toolbar.append(copy);
      pre.replaceWith(wrapper);
      wrapper.append(toolbar, pre);
      if (language === "language-plotly") renderPlotlyBlock(code.textContent, wrapper);
    });
  }

  function renderPlotlyBlock(source, wrapper) {
    const chart = document.createElement("div");
    chart.className = "plotly-chart";
    chart.setAttribute("role", "img");
    chart.setAttribute("aria-label", "Interactive Plotly chart");
    wrapper.append(chart);
    try {
      if (source.length > 100_000) throw new Error("The chart definition is too large.");
      if (!window.Plotly) throw new Error("Plotly could not be loaded.");
      const figure = JSON.parse(source);
      const rawTraces = Array.isArray(figure.data) ? figure.data.slice(0, 20) : [];
      const allowedTypes = new Set(["scatter", "scattergl", "bar", "pie"]);
      const traces = rawTraces.map((item) => {
        if (!item || typeof item !== "object" || !allowedTypes.has(item.type)) {
          throw new Error("Plotly blocks need supported JSON chart traces.");
        }
        const trace = { type: item.type };
        for (const key of ["x", "y", "labels", "values"]) {
          if (Array.isArray(item[key]) && item[key].length <= 5000
            && item[key].every((value) => ["string", "number"].includes(typeof value) || value === null)) {
            trace[key] = item[key];
          }
        }
        if (typeof item.name === "string") trace.name = item.name.slice(0, 100);
        if (item.type === "scatter" || item.type === "scattergl") {
          trace.mode = ["lines", "markers", "lines+markers"].includes(item.mode) ? item.mode : "lines";
        }
        if (item.marker && typeof item.marker === "object") {
          const marker = {};
          if (typeof item.marker.color === "string") marker.color = item.marker.color.slice(0, 80);
          if (typeof item.marker.size === "number") marker.size = Math.min(24, Math.max(2, item.marker.size));
          if (Object.keys(marker).length) trace.marker = marker;
        }
        return trace;
      });
      if (!traces.length) throw new Error("No supported chart traces were provided.");
      const rawLayout = figure.layout && typeof figure.layout === "object" ? figure.layout : {};
      const layout = {
        template: "plotly_dark",
        autosize: true,
        margin: { l: 45, r: 15, t: 40, b: 40 },
        showlegend: rawLayout.showlegend !== false,
      };
      if (typeof rawLayout.title === "string") layout.title = rawLayout.title.slice(0, 120);
      for (const axisName of ["xaxis", "yaxis"]) {
        const axis = rawLayout[axisName];
        if (!axis || typeof axis !== "object") continue;
        layout[axisName] = {};
        if (typeof axis.title === "string") layout[axisName].title = axis.title.slice(0, 100);
        if (typeof axis.type === "string" && ["linear", "log", "date", "category"].includes(axis.type)) {
          layout[axisName].type = axis.type;
        }
      }
      Plotly.newPlot(chart, traces, layout, { responsive: true, displaylogo: false })
        .catch((error) => {
          chart.textContent = `Could not render chart: ${error.message}`;
          chart.classList.add("plotly-error");
        });
    } catch (error) {
      chart.textContent = `Could not render chart: ${error.message}`;
      chart.classList.add("plotly-error");
    }
  }

  async function runCode(source, wrapper, button) {
    button.disabled = true;
    let output = $(".code-output", wrapper);
    if (!output) {
      output = document.createElement("pre");
      output.className = "code-output";
      output.setAttribute("aria-live", "polite");
      wrapper.append(output);
    }
    output.textContent = "Running restricted Python…";
    try {
      const { response } = await api("/api/run-code", {
        method: "POST",
        body: JSON.stringify({ code: source }),
      });
      const result = await response.json();
      output.textContent = result.output || "Finished without output.";
      showToast("Code ran in the restricted local interpreter.");
    } catch (error) {
      output.textContent = `Run blocked: ${error.message}`;
      showToast(error.message, true);
    } finally {
      button.disabled = !state.executionEnabled;
    }
  }

  async function copyText(value, successText) {
    try {
      await navigator.clipboard.writeText(value);
      showToast(successText);
    } catch (_error) {
      const input = document.createElement("textarea");
      input.value = value;
      input.style.position = "fixed";
      input.style.opacity = "0";
      document.body.append(input);
      input.select();
      const copied = document.execCommand("copy");
      input.remove();
      showToast(copied ? successText : "Could not access the clipboard.", !copied);
    }
  }

  function setTheme(theme) {
    const next = theme === "light" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    document.querySelector('meta[name="theme-color"]').content = next === "dark" ? "#0f1117" : "#f7f8fb";
    localStorage.setItem("theme", next);
    elements.themeIcon.textContent = next === "dark" ? "☼" : "☾";
    elements.themeLabel.textContent = next === "dark" ? "Light appearance" : "Dark appearance";
  }

  function applyStatus(status) {
    state.online = Boolean(status.online);
    state.provider = status.provider || "local";
    state.executionEnabled = Boolean(status.execution_enabled);
    state.visionModel = status.vision_model || "qwen2.5vl:7b";
    const availableModels = Array.isArray(status.models) ? status.models : [];
    if (availableModels.length) {
      const previous = state.model || status.model;
      elements.model.replaceChildren();
      availableModels.forEach((model) => {
        const option = document.createElement("option");
        option.value = model;
        option.textContent = model;
        elements.model.append(option);
      });
      const base = (value) => String(value || "").split(":")[0];
      state.model = availableModels.includes(previous)
        ? previous
        : availableModels.find((model) => base(model) === base(status.model)) || availableModels[0];
      elements.model.value = state.model;
    } else if (status.model && !elements.model.options.length) {
      const option = document.createElement("option");
      option.value = status.model;
      option.textContent = status.model;
      elements.model.add(option);
      state.model = status.model;
    } else if (status.model && !state.model) {
      state.model = status.model;
    }
    const dot = $(".status-dot", elements.engineBadge);
    dot.classList.toggle("online", state.online);
    dot.classList.toggle("offline", !state.online);
    const welcomeDot = $(".status-dot", elements.welcome);
    welcomeDot.classList.toggle("online", state.online);
    welcomeDot.classList.toggle("offline", !state.online);
    const label = state.online
      ? `${capitalize(state.provider)} · ${state.model || status.model || "ready"}`
      : `${capitalize(state.provider)} unavailable`;
    elements.engineLabel.textContent = label;
    elements.welcomeStatus.textContent = state.online ? `${state.model} · Ready on ${state.provider}` : "Ollama is offline";
    elements.providerHint.textContent = status.public_mode
      ? "Public server · code runner disabled"
      : state.provider === "local"
        ? "Private · runs on your device"
        : "Cloud provider configured";
    $$(".run-code").forEach((button) => {
      button.disabled = !state.executionEnabled;
      button.title = state.executionEnabled
        ? "Run in the restricted local Python interpreter"
        : "Disabled in public/cloud mode";
    });
  }

  async function refreshStatus() {
    try {
      const { response } = await api("/api/status");
      applyStatus(await response.json());
    } catch (error) {
      state.online = false;
      applyStatus({ online: false, provider: state.provider, model: state.model });
      showToast(`Could not check AI status: ${error.message}`, true);
    }
  }

  async function refreshHistory() {
    try {
      const { response } = await api("/api/chats");
      const data = await response.json();
      renderHistory(data.chats || []);
    } catch (error) {
      showToast(`Could not load conversations: ${error.message}`, true);
    }
  }

  function renderHistory(chats) {
    elements.history.replaceChildren();
    if (!chats.length) {
      const empty = document.createElement("div");
      empty.className = "conversation-empty";
      empty.textContent = "Your saved conversations will appear here.";
      elements.history.append(empty);
      return;
    }
    chats.forEach((chat) => {
      const row = document.createElement("div");
      row.className = "conversation-row";
      const button = document.createElement("button");
      button.className = `conversation-link${chat.id === state.conversationId ? " active" : ""}`;
      button.type = "button";
      button.textContent = chat.title || "Conversation";
      button.title = button.textContent;
      button.addEventListener("click", () => loadConversation(chat.id));
      const rename = document.createElement("button");
      rename.className = "conversation-action";
      rename.type = "button";
      rename.textContent = "✎";
      rename.title = "Rename conversation";
      rename.setAttribute("aria-label", `Rename ${chat.title}`);
      rename.addEventListener("click", () => openRename(chat));
      const remove = document.createElement("button");
      remove.className = "conversation-action delete";
      remove.type = "button";
      remove.textContent = "×";
      remove.title = "Delete conversation";
      remove.setAttribute("aria-label", `Delete ${chat.title}`);
      remove.addEventListener("click", () => deleteConversation(chat));
      row.append(button, rename, remove);
      elements.history.append(row);
    });
  }

  async function loadConversation(id) {
    if (state.busy) return showToast("Wait for the current response to finish.");
    try {
      const { response } = await api(`/api/chats/${encodeURIComponent(id)}`);
      const { chat } = await response.json();
      state.conversationId = chat.id;
      state.model = chat.model || state.model;
      setTitle(chat.title || "Conversation");
      elements.persona.value = chat.persona || "Academic Tutor";
      elements.messages.replaceChildren();
      hideWelcome();
      (chat.messages || []).forEach((item) => {
        createMessage(item.role, item.display_content || item.content || "", {
          sources: item.sources || [],
        });
      });
      if (!(chat.messages || []).length) elements.welcome.hidden = false;
      await refreshHistory();
      closeDrawer();
      scrollToLatest(false);
    } catch (error) {
      showToast(`Could not open conversation: ${error.message}`, true);
    }
  }

  function openRename(chat) {
    state.renameId = chat.id;
    elements.renameInput.value = chat.title || "";
    elements.renameDialog.showModal();
    elements.renameInput.focus();
    elements.renameInput.select();
  }

  async function renameConversation() {
    const title = elements.renameInput.value.trim();
    if (!title) return showToast("Enter a conversation name.", true);
    try {
      await api(`/api/chats/${encodeURIComponent(state.renameId)}`, {
        method: "PATCH",
        body: JSON.stringify({ title }),
      });
      if (state.renameId === state.conversationId) setTitle(title);
      elements.renameDialog.close();
      await refreshHistory();
      showToast("Conversation renamed");
    } catch (error) {
      showToast(`Could not rename: ${error.message}`, true);
    }
  }

  async function deleteConversation(chat) {
    if (!window.confirm(`Delete “${chat.title || "Conversation"}” from this browser's local history?`)) return;
    try {
      await api(`/api/chats/${encodeURIComponent(chat.id)}`, { method: "DELETE" });
      if (chat.id === state.conversationId) newConversation();
      await refreshHistory();
      showToast("Conversation deleted");
    } catch (error) {
      showToast(`Could not delete: ${error.message}`, true);
    }
  }

  function newConversation() {
    if (state.busy) return showToast("Wait for the current response to finish.");
    state.conversationId = newId();
    setTitle("New conversation");
    elements.messages.replaceChildren();
    elements.welcome.hidden = false;
    elements.input.value = "";
    state.attachments = [];
    renderAttachmentPills();
    refreshHistory();
    closeDrawer();
    elements.input.focus();
  }

  async function exportConversation() {
    if (state.title === "New conversation" && !elements.messages.children.length) {
      return showToast("Send a message before exporting this conversation.");
    }
    try {
      const { response } = await api(`/api/chats/${encodeURIComponent(state.conversationId)}/export.md`);
      const blob = await response.blob();
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `${safeFilename(state.title)}.md`;
      link.click();
      URL.revokeObjectURL(link.href);
      showToast("Markdown export downloaded");
    } catch (error) {
      showToast(`Could not export: ${error.message}`, true);
    }
  }

  function safeFilename(value) {
    return value.replace(/[^\p{L}\p{N}_-]+/gu, "-").replace(/^-|-$/g, "").slice(0, 60) || "conversation";
  }

  async function attachFiles(fileList) {
    const files = [...fileList];
    if (!files.length) return;
    if (files.length + state.attachments.length > 5) {
      showToast("Attach up to five documents and images at a time.", true);
      return;
    }
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    const pill = document.createElement("div");
    pill.className = "upload-pill";
    pill.textContent = "Reading documents…";
    elements.pills.append(pill);
    try {
      const { response } = await api("/upload", { method: "POST", body: form });
      const data = await response.json();
      state.attachments.push(...data.files);
      renderAttachmentPills();
      showToast(`${data.files.length} file(s) attached`);
    } catch (error) {
      pill.remove();
      showToast(`Could not read document: ${error.message}`, true);
    } finally {
      elements.attach.value = "";
    }
  }

  function renderAttachmentPills() {
    elements.pills.replaceChildren();
    state.attachments.forEach((file, index) => {
      const pill = document.createElement("div");
      pill.className = "upload-pill";
      const label = document.createElement("span");
      if (file.kind === "image") {
        const preview = document.createElement("img");
        preview.src = `data:${file.mime_type};base64,${file.data}`;
        preview.alt = "";
        pill.append(preview);
        label.textContent = `▧ ${file.name}`;
      } else {
        label.textContent = `▤ ${file.name}${file.truncated ? " · shortened" : ""}`;
      }
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "×";
      remove.setAttribute("aria-label", `Remove ${file.name}`);
      remove.addEventListener("click", () => {
        state.attachments.splice(index, 1);
        renderAttachmentPills();
      });
      pill.append(label, remove);
      elements.pills.append(pill);
    });
  }

  async function submitMessage(event) {
    event?.preventDefault();
    if (state.busy) return;
    const typed = elements.input.value.trim();
    if (!typed && !state.attachments.length) return;
    const attachments = state.attachments.splice(0);
    renderAttachmentPills();
    let prompt = typed;
    const docs = attachments.filter((file) => file.kind === "document");
    const images = attachments.filter((file) => file.kind === "image");
    if (docs.length) {
      const context = selectDocumentChunks(docs, typed);
      prompt = `${typed || "Please analyze these documents."}\n\n${context}`.slice(0, 80000);
    }
    if (!typed && images.length) prompt = "Please analyze the attached image(s).";
    const attachmentLabels = attachments.map((file) => file.name);
    const displayMessage = [
      typed || (images.length ? "Please analyze the attached image(s)." : "Please analyze the attached document(s)."),
      ...(attachmentLabels.length ? [`Attachments: ${attachmentLabels.join(", ")}`] : []),
    ].join("\n");
    elements.input.value = "";
    resizeInput();
    createMessage("user", displayMessage, { images });
    const assistant = createMessage("assistant", "", { thinking: true });
    setBusy(true);
    const messages = await streamChat({
      conversation_id: state.conversationId,
      message: prompt,
      display_message: displayMessage,
      model: state.model,
      persona: elements.persona.value,
      style: elements.responseStyle.value,
      web_search: elements.webSearch.checked,
      search_query: typed || attachmentLabels.join(" "),
      images: images.map(({ name, mime_type, data }) => ({ name, mime_type, data })),
    }, assistant);
    setBusy(false);
    if (messages?.done) {
      state.conversationId = messages.done.conversation_id;
      setTitle(messages.done.title || state.title);
      state.provider = messages.done.provider || state.provider;
      state.model = messages.done.model || state.model;
      if ([...elements.model.options].some((option) => option.value === state.model)) {
        elements.model.value = state.model;
      }
      refreshHistory();
    }
    elements.input.focus();
  }

  async function streamChat(payload, assistant) {
    let answer = "";
    try {
      const { response } = await api("/api/chat/stream", {
        method: "POST",
        body: JSON.stringify(payload),
        headers: { Accept: "text/event-stream" },
      });
      if (!response.body) throw new Error("Streaming is unavailable in this browser.");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let pending = "";
      let finalEvent = null;
      let sources = [];
      while (true) {
        const { value, done } = await reader.read();
        pending += decoder.decode(value || new Uint8Array(), { stream: !done });
        const chunks = pending.split(/\r?\n\r?\n/);
        pending = chunks.pop() || "";
        for (const chunk of chunks) {
          const parsed = parseSseChunk(chunk);
          if (!parsed) continue;
          if (parsed.event === "token") {
            answer += parsed.data.token || "";
            assistant.card.textContent = answer;
            scrollToLatest();
          } else if (parsed.event === "done") {
            sources = parsed.data.sources || sources;
            finalEvent = { done: parsed.data };
          } else if (parsed.event === "sources") {
            sources = parsed.data.sources || [];
            if (!sources.length) {
              showToast("Web Search found no sources; continuing without citations.");
            }
          } else if (parsed.event === "error") {
            throw new Error(parsed.data.message || "AI request failed.");
          }
        }
        if (done) break;
      }
      if (!answer && !finalEvent) throw new Error("The AI ended the stream without a response.");
      renderMarkdown(assistant.card, answer || "No response was returned.");
      addMessageTools(assistant.content, answer);
      if (sources.length) addCitations(assistant.content, sources);
      if (finalEvent?.done) finalEvent.done.sources = sources;
      return finalEvent;
    } catch (error) {
      assistant.card.textContent = `Unable to complete the request. ${error.message}`;
      assistant.card.classList.add("error-card");
      showToast(error.message, true, 5200);
      return null;
    }
  }

  function parseSseChunk(chunk) {
    if (!chunk.trim()) return null;
    let event = "message";
    const dataLines = [];
    chunk.split(/\r?\n/).forEach((line) => {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    });
    try {
      return { event, data: JSON.parse(dataLines.join("\n")) };
    } catch (_error) {
      return null;
    }
  }

  function resizeInput() {
    elements.input.style.height = "auto";
    elements.input.style.height = `${Math.min(elements.input.scrollHeight, 190)}px`;
  }

  function useStarter(prompt) {
    elements.input.value = prompt;
    resizeInput();
    elements.input.focus();
    elements.input.setSelectionRange(prompt.length, prompt.length);
  }

  function capitalize(value) {
    return value ? value[0].toUpperCase() + value.slice(1) : "AI";
  }

  function cleanForSpeech(value) {
    return String(value)
      .replace(/```[\w-]*\s*([\s\S]*?)```/g, "$1")
      .replace(/!\[([^\]]*)\]\([^)]+\)/g, "$1")
      .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
      .replace(/\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}/g, "$1 divided by $2")
      .replace(/\\sqrt\s*\{([^{}]*)\}/g, "square root of $1")
      .replace(/\\(alpha|beta|gamma|delta|theta|lambda|mu|pi|sigma|omega)\b/gi, "$1")
      .replace(/\\[a-zA-Z]+/g, " ")
      .replace(/\\(?:\(|\)|\[|\])/g, " ")
      .replace(/(\*\*|__|~~|`|[*_#$^{}])/g, " ")
      .replace(/^\s{0,3}(?:#{1,6}\s|>\s|[-*+]\s|\d+\.\s)/gm, "")
      .replace(/\s+/g, " ")
      .trim();
  }

  function speakText(text) {
    if (!("speechSynthesis" in window)) {
      return showToast("Read aloud is not supported by this browser.", true);
    }
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(cleanForSpeech(text));
    utterance.rate = Number(elements.speechRate.value) || 1;
    const selectedVoice = window.speechSynthesis.getVoices()
      .find((voice) => voice.voiceURI === elements.voiceSelect.value);
    if (selectedVoice) utterance.voice = selectedVoice;
    window.speechSynthesis.speak(utterance);
  }

  function populateVoices() {
    if (!("speechSynthesis" in window)) return;
    const voices = window.speechSynthesis.getVoices();
    const previous = localStorage.getItem("speechVoice") || "";
    elements.voiceSelect.replaceChildren(new Option("System default", ""));
    voices.forEach((voice) => {
      const option = new Option(`${voice.name} · ${voice.lang}`, voice.voiceURI);
      elements.voiceSelect.add(option);
    });
    if ([...elements.voiceSelect.options].some((option) => option.value === previous)) {
      elements.voiceSelect.value = previous;
    }
  }

  function toggleVoice() {
    state.voiceEnabled = !state.voiceEnabled;
    localStorage.setItem("voiceEnabled", String(state.voiceEnabled));
    elements.voiceToggle.classList.toggle("active", state.voiceEnabled);
    elements.voiceToggle.title = state.voiceEnabled ? "Read aloud enabled" : "Read aloud disabled";
    showToast(state.voiceEnabled ? "Read aloud enabled" : "Read aloud disabled");
  }

  function initSpeechRecognition() {
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) {
      elements.dictate.title = "Voice dictation is not supported in this browser";
      elements.dictate.addEventListener("click", () => {
        showToast("Voice dictation is not supported by this browser. Try a current Chrome or Safari.", true);
      });
      return;
    }
    state.recognition = new Recognition();
    state.recognition.lang = navigator.language || "en-US";
    state.recognition.interimResults = true;
    state.recognition.continuous = false;
    let finalText = "";
    state.recognition.onstart = () => {
      elements.dictate.classList.add("listening");
      elements.dictate.setAttribute("aria-label", "Listening, click to stop");
      showToast("Listening… Speak your message.");
    };
    state.recognition.onresult = (event) => {
      let interim = "";
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const transcript = event.results[index][0].transcript;
        if (event.results[index].isFinal) finalText += transcript;
        else interim += transcript;
      }
      const prefix = elements.input.dataset.beforeDictation || "";
      elements.input.value = `${prefix}${prefix && (finalText || interim) ? " " : ""}${finalText || interim}`;
      resizeInput();
    };
    state.recognition.onerror = (event) => {
      const reason = event.error === "not-allowed"
        ? "Microphone permission denied. Allow microphone access in browser settings."
        : event.error === "no-speech"
          ? "No speech detected. Try again."
          : `Voice input failed: ${event.error}.`;
      showToast(reason, true);
    };
    state.recognition.onend = () => {
      elements.dictate.classList.remove("listening");
      elements.dictate.setAttribute("aria-label", "Dictate message");
      elements.input.dataset.beforeDictation = "";
      elements.input.focus();
    };
    elements.dictate.addEventListener("click", () => {
      if (elements.dictate.classList.contains("listening")) {
        state.recognition.stop();
        return;
      }
      finalText = "";
      elements.input.dataset.beforeDictation = elements.input.value.trim();
      try {
        state.recognition.start();
      } catch (_error) {
        showToast("Voice input is already active.", true);
      }
    });
  }

  function setupDrawer() {
    $("#open-drawer").addEventListener("click", () => document.body.classList.add("drawer-open"));
    $("#close-drawer").addEventListener("click", closeDrawer);
    $("#scrim").addEventListener("click", closeDrawer);
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeDrawer();
    });
  }

  function closeDrawer() {
    document.body.classList.remove("drawer-open");
  }

  async function registerPwa() {
    if ("serviceWorker" in navigator) {
      try {
        await navigator.serviceWorker.register("/sw.js", { scope: "/" });
      } catch (_error) {
        showToast("Offline shell could not be cached on this device.");
      }
    }
    window.addEventListener("beforeinstallprompt", (event) => {
      event.preventDefault();
      state.deferredInstall = event;
      elements.install.hidden = false;
    });
    elements.install.addEventListener("click", async () => {
      if (!state.deferredInstall) {
        showToast("Use your browser’s Share or menu button and choose “Add to Home Screen”.");
        return;
      }
      state.deferredInstall.prompt();
      await state.deferredInstall.userChoice;
      state.deferredInstall = null;
      elements.install.hidden = true;
    });
    window.addEventListener("appinstalled", () => {
      elements.install.hidden = true;
      showToast("All-in-One AI Studio installed");
    });
  }

  function bindEvents() {
    elements.form.addEventListener("submit", submitMessage);
    elements.input.addEventListener("input", resizeInput);
    elements.input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        submitMessage(event);
      }
    });
    elements.attach.addEventListener("change", () => attachFiles(elements.attach.files));
    elements.themeToggle.addEventListener("click", () => {
      setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
    });
    elements.voiceToggle.addEventListener("click", toggleVoice);
    elements.voiceSelect.addEventListener("change", () => {
      localStorage.setItem("speechVoice", elements.voiceSelect.value);
    });
    elements.speechRate.addEventListener("input", () => {
      elements.speechRateValue.value = `${Number(elements.speechRate.value).toFixed(1)}×`;
      localStorage.setItem("speechRate", elements.speechRate.value);
    });
    elements.model.addEventListener("change", () => { state.model = elements.model.value; });
    $("#new-chat").addEventListener("click", newConversation);
    $("#export-button").addEventListener("click", exportConversation);
    $("#refresh-status").addEventListener("click", () => {
      refreshStatus();
      showToast("Refreshing AI status…");
    });
    $("#refresh-history").addEventListener("click", refreshHistory);
    $("#rename-confirm").addEventListener("click", async (event) => {
      event.preventDefault();
      await renameConversation();
    });
    $("#rename-form").addEventListener("submit", (event) => event.preventDefault());
    $$(".starter-card").forEach((button) => {
      button.addEventListener("click", () => useStarter(button.dataset.prompt));
    });
    $$(".tool-button").forEach((button) => {
      button.addEventListener("click", () => useStarter(button.dataset.prompt));
    });
    window.addEventListener("online", refreshStatus);
    window.addEventListener("offline", () => showToast("This device is offline.", true));
  }

  async function init() {
    setTheme(localStorage.getItem("theme") || "dark");
    elements.speechRate.value = localStorage.getItem("speechRate") || "1";
    elements.speechRateValue.value = `${Number(elements.speechRate.value).toFixed(1)}×`;
    populateVoices();
    if ("speechSynthesis" in window) {
      window.speechSynthesis.addEventListener("voiceschanged", populateVoices);
    }
    elements.voiceToggle.classList.toggle("active", state.voiceEnabled);
    setupDrawer();
    bindEvents();
    initSpeechRecognition();
    await registerPwa();
    await Promise.all([refreshStatus(), refreshHistory()]);
    state.statusTimer = setInterval(refreshStatus, 60_000);
    elements.input.focus({ preventScroll: true });
  }

  init();
})();
