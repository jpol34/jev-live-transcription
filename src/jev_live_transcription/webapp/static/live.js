// Matches config.TICK_SECONDS (1) -- duplicated here since the frontend has no server-rendered
// template step to source it from live.
const TICK_SECONDS = 1;

const FIELD_LABELS = {
  caller_name: "Name",
  email: "Email",
  phone_number: "Phone",
  unit_number: "Unit",
  amenities_requested: "Amenities",
  pet_info: "Pet Info",
  permission_to_enter: "Permission to Enter",
  work_order_issue: "Issue",
  move_in_date: "Move-in Date",
  price_quoted: "Price Quoted",
  budget_amount: "Budget",
};

const FORM_TEMPLATES = {
  prospect: {
    title: "Guest Card",
    subtitle: "Prospect",
    fields: [
      "caller_name",
      "email",
      "phone_number",
      "unit_number",
      "amenities_requested",
      "pet_info",
      "move_in_date",
      "price_quoted",
      "budget_amount",
    ],
  },
  resident: {
    title: "Work Order",
    subtitle: "Resident",
    fields: ["caller_name", "phone_number", "unit_number", "work_order_issue", "permission_to_enter"],
  },
};

const callId = window.location.pathname.split("/").filter(Boolean).pop();

const el = {
  callLabel: document.getElementById("call-label"),
  tickNumber: document.getElementById("tick-number"),
  totalTicks: document.getElementById("total-ticks"),
  elapsed: document.getElementById("elapsed"),
  callerTypePill: document.getElementById("caller-type-pill"),
  transcript: document.getElementById("transcript"),
  form: document.getElementById("form"),
  banner: document.getElementById("banner"),
};

el.callLabel.textContent = `Call ${callId}`;

function fieldRow(label, entry) {
  const row = document.createElement("div");
  row.className = "field-row";
  const labelSpan = document.createElement("span");
  labelSpan.className = "label";
  labelSpan.textContent = label;
  const valueSpan = document.createElement("span");
  if (entry && entry.value) {
    valueSpan.className = "value";
    valueSpan.textContent = entry.value;
  } else {
    valueSpan.className = "value empty";
    valueSpan.textContent = "not yet mentioned";
  }
  row.appendChild(labelSpan);
  row.appendChild(valueSpan);
  return row;
}

function renderForm(callerType, committed) {
  const glinerJev = committed.gliner_jev || {};
  el.form.innerHTML = "";

  const template = FORM_TEMPLATES[callerType];
  const titleEl = document.createElement("p");
  titleEl.className = "form-title";
  const subtitleEl = document.createElement("p");
  subtitleEl.className = "form-subtitle";

  if (!template) {
    // "pending" (not yet classified) or "other" -- no fixed form shape for either, so show
    // whatever's been extracted so far as a plain list rather than a structured template.
    titleEl.textContent = callerType === "other" ? "General Message" : "Determining caller type…";
    subtitleEl.textContent = callerType === "other" ? "Other" : "";
    el.form.appendChild(titleEl);
    el.form.appendChild(subtitleEl);
    const anyFields = Object.keys(FIELD_LABELS).filter((field) => glinerJev[field] && glinerJev[field].value);
    if (anyFields.length === 0) {
      const empty = document.createElement("p");
      empty.className = "form-subtitle";
      empty.textContent = "Nothing extracted yet.";
      el.form.appendChild(empty);
    } else {
      for (const field of anyFields) {
        el.form.appendChild(fieldRow(FIELD_LABELS[field], glinerJev[field]));
      }
    }
    return;
  }

  titleEl.textContent = template.title;
  subtitleEl.textContent = template.subtitle;
  el.form.appendChild(titleEl);
  el.form.appendChild(subtitleEl);
  for (const field of template.fields) {
    el.form.appendChild(fieldRow(FIELD_LABELS[field], glinerJev[field]));
  }
}

function renderTranscript(text) {
  el.transcript.innerHTML = "";
  for (const line of text.split("\n")) {
    if (!line) continue;
    const [speaker, ...rest] = line.split(": ");
    const lineEl = document.createElement("div");
    if (rest.length > 0 && (speaker === "Agent" || speaker === "Caller")) {
      const speakerSpan = document.createElement("span");
      speakerSpan.className = `speaker-${speaker}`;
      speakerSpan.textContent = `${speaker}: `;
      lineEl.appendChild(speakerSpan);
      lineEl.appendChild(document.createTextNode(rest.join(": ")));
    } else {
      lineEl.textContent = line;
    }
    el.transcript.appendChild(lineEl);
  }
  el.transcript.scrollTop = el.transcript.scrollHeight;
}

function showBanner(className, text) {
  el.banner.innerHTML = "";
  const banner = document.createElement("div");
  banner.className = className;
  banner.textContent = text;
  el.banner.appendChild(banner);
}

function connect() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${protocol}://${window.location.host}/ws/${callId}`);
  // Tracks whether the connection ever reached a terminal state on its own (a "busy"/"done"
  // message, or an onerror) -- onclose fires for *every* disconnect, including those, so it must
  // not show a redundant/contradictory "connection lost" banner over an already-shown one.
  let settled = false;

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);

    if (msg.type === "busy") {
      settled = true;
      showBanner("busy-banner", msg.message || "Busy -- try again shortly.");
      return;
    }
    if (msg.type === "done") {
      settled = true;
      showBanner("done-banner", "Call finished.");
      return;
    }

    el.tickNumber.textContent = msg.tick_number;
    el.totalTicks.textContent = msg.total_ticks;
    el.elapsed.textContent = `${msg.tick_number * TICK_SECONDS}s`;
    renderTranscript(msg.transcript || "");

    const status = msg.caller_type ? msg.caller_type.status : "pending";
    el.callerTypePill.textContent = status;
    el.callerTypePill.className = status === "pending" ? "pill pending" : "pill";
    renderForm(status, msg.committed || {});
  };

  ws.onerror = () => {
    settled = true;
    showBanner("busy-banner", "Connection error -- try refreshing the page.");
  };

  ws.onclose = () => {
    // A clean server-side close with no preceding message (e.g. an unknown call_id, or an
    // internal error before any tick was sent) fires only this handler, not onmessage/onerror --
    // without it, the page would otherwise sit on "Loading..."/tick 0 forever with no feedback.
    if (!settled) {
      showBanner("busy-banner", "Connection closed -- try refreshing the page.");
    }
  };
}

connect();
