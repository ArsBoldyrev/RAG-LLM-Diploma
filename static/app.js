const chatEl = document.getElementById("chat");
const msgInput = document.getElementById("message");
const sendBtn = document.getElementById("send");

function addBubble(text, who = "user") {
  const wrap = document.createElement("div");
  wrap.className = `flex ${who === "user" ? "justify-end" : "justify-start"}`;

  const bubble = document.createElement("div");
  bubble.className = `${
    who === "user" ? "bg-sky-500 text-white" : "bg-gray-700 text-gray-100"
  } rounded-2xl px-4 py-3 max-w-[85%] whitespace-pre-wrap leading-6`;
  if (text) bubble.textContent = text;

  wrap.appendChild(bubble);
  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
  return bubble;
}

function mdEscape(s) {
  return s.replace(/[&<>]/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[ch]));
}

function renderSources(container, sources) {
  if (!sources || !sources.length) return;

  const sep = document.createElement("div");
  sep.className = "text-sm text-gray-300 mt-3 mb-1";
  sep.textContent = "Источники:";
  container.appendChild(sep);

  const badges = document.createElement("div");
  badges.className = "flex flex-wrap gap-2";
  container.appendChild(badges);

  const snippets = document.createElement("div");
  snippets.className = "mt-2 space-y-2";
  container.appendChild(snippets);

  sources.forEach(src => {
    const btn = document.createElement("button");
    btn.className = "text-xs px-2 py-1 rounded bg-gray-600 hover:bg-gray-500";
    btn.textContent = `[${src.label}]`;
    btn.title = `${src.file}, стр. ${src.page}`;
    btn.addEventListener("click", () => {
      const id = `snip-${src.label}-${src.id}`;
      let sn = document.getElementById(id);
      if (sn) { sn.remove(); return; }
      sn = document.createElement("div");
      sn.id = id;
      sn.className = "text-xs bg-gray-800 border border-gray-700 rounded p-2";
      sn.innerHTML = `<div class="font-semibold mb-1">[${src.label}] ${mdEscape(src.file)}, стр. ${src.page}</div><div>${mdEscape(src.snippet || "")}</div>`;
      snippets.appendChild(sn);
      chatEl.scrollTop = chatEl.scrollHeight;
    });
    badges.appendChild(btn);
  });
}

async function askStream() {
  const q = msgInput.value.trim();
  if (!q) return;

  // пузырь пользователя
  addBubble(q, "user");
  msgInput.value = "";

  // пузырь ассистента
  const assistantBubble = addBubble("", "assistant");

  // якорь источников
  const sourcesAnchor = document.createElement("div");
  assistantBubble.appendChild(sourcesAnchor);

  // контейнер для текста ответа
  const content = document.createElement("div");
  content.className = "answer-stream";
  content.style.whiteSpace = "pre-wrap";
  content.style.lineHeight = "1.6";
  content.style.fontSize = "1rem";
  content.style.color = "rgb(243,244,246)";
  assistantBubble.insertBefore(content, sourcesAnchor);

  // индикатор «печатает…»
  const typing = document.createElement("span");
  typing.className = "ml-2 opacity-70";
  typing.textContent = "▍";
  assistantBubble.insertBefore(typing, sourcesAnchor);
  const blink = setInterval(() => {
    typing.style.opacity = typing.style.opacity === "0.2" ? "0.7" : "0.2";
  }, 400);

  // собираем форму
  const form = new FormData();
  form.append("question", q);
  form.append("k", document.getElementById("k").value);
  form.append("fetch_k", document.getElementById("fetch_k").value);
  form.append("use_reranker", document.getElementById("use_reranker").checked ? "true" : "false");
  form.append("reranker_model", document.getElementById("reranker_model").value);
  form.append("llm_model", document.getElementById("llm_model").value);
  form.append("temperature", document.getElementById("temperature").value);

  try {
    const res = await fetch("/api/chat/stream", { method: "POST", body: form });
    if (!res.ok || !res.body) {
      content.textContent += "\nОшибка: не удалось открыть поток.";
      clearInterval(blink); typing.remove();
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");

    // -------- НОВЫЙ ПАРСЕР: флашим по встрече `event:` ИЛИ по пустой строке --------
    let currentEvent = "message";
    let dataLines = [];
    let pending = "";

    const flush = () => {
      if (dataLines.length === 0 && currentEvent === "message") return;
      const dataRaw = dataLines.join("\n");
      dataLines = [];

      let payload = null;
      try { payload = JSON.parse(dataRaw); } catch {}

      if (currentEvent === "sources") {
        renderSources(assistantBubble, (payload && payload.sources) ? payload.sources : []);
      } else if (currentEvent === "token") {
        const text = (payload && typeof payload.message === "string") ? payload.message : dataRaw;
        content.textContent += text;
        chatEl.scrollTop = chatEl.scrollHeight;
      } else if (currentEvent === "done") {
        clearInterval(blink); typing.remove();
      }
      // после флаша оставим текущее имя события как есть — сервер может прислать подряд несколько data для того же event
    };

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      pending += decoder.decode(value, { stream: true });

      // читаем построчно
      let nl;
      while ((nl = pending.indexOf("\n")) >= 0) {
        let line = pending.slice(0, nl);
        pending = pending.slice(nl + 1);
        if (line.endsWith("\r")) line = line.slice(0, -1);

        if (line === "") {
          // классический конец события
          flush();
          currentEvent = "message";
          continue;
        }
        if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).trim());
          continue;
        }
        if (line.startsWith("event:")) {
          // если уже накоплены data для прошлого события — флашим их ПРЕЖДЕ чем переключить event
          if (dataLines.length > 0) flush();
          currentEvent = line.slice(6).trim() || "message";
          continue;
        }
        // остальные поля игнорируем
      }
    }

    // добить хвост (если сервер закрыл поток без пустой строки)
    if (dataLines.length > 0) flush();
    clearInterval(blink); typing.remove();

  } catch (e) {
    clearInterval(blink); typing.remove();
    content.textContent += "\nОшибка запроса: " + e;
  }
}

sendBtn.addEventListener("click", askStream);
msgInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    askStream();
  }
});


sendBtn.addEventListener("click", askStream);
msgInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    askStream();
  }
});

// загрузка PDF как было
document.getElementById("upload").addEventListener("click", async () => {
  const files = document.getElementById("files").files;
  if (!files.length) return;
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  const statusEl = document.getElementById("upload-status");
  statusEl.textContent = "Идёт загрузка…";
  try {
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    const data = await res.json();
    statusEl.textContent = data.status === "ok"
      ? `Готово. Добавлено чанков: ${data.indexed_chunks}`
      : JSON.stringify(data);
  } catch (e) {
    statusEl.textContent = "Ошибка: " + e;
  }
});
