(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const labels = {
    SUPPORTED: { text: "Supported", className: "supported" },
    CONTRADICTED: { text: "Contradicted", className: "contradicted" },
    INSUFFICIENT_EVIDENCE: { text: "Insufficient evidence", className: "insufficient" },
    CONFLICTING_EVIDENCE: { text: "Conflicting evidence", className: "conflicting" },
  };
  const form = $("verify-form");
  const evidenceFields = $("evidence-fields");
  const pane = document.querySelector(".result-pane");
  let passageId = 0;

  const searching = () => form.elements.mode.value === "search";

  function updateMode() {
    $("own-evidence").hidden = searching();
    $("own-evidence").disabled = searching();
    $("kb-note").hidden = !searching();
    $("evidence-list-heading").textContent = searching()
      ? "Visa passages found" : "Compare supplied evidence";
    resetResults();
  }

  function updatePassages() {
    const rows = [...evidenceFields.children];
    rows.forEach((row, index) => {
      row.querySelector(".passage-label").textContent = `Passage ${index + 1}`;
      const remove = row.querySelector(".remove-passage");
      remove.hidden = rows.length === 1;
      remove.setAttribute("aria-label", `Remove passage ${index + 1}`);
    });
    $("passage-count").textContent = `${rows.length} of 20`;
    $("add-evidence").disabled = rows.length >= 20;
  }

  function addPassage(text = "", source = "", focus = false) {
    const row = $("evidence-template").content.firstElementChild.cloneNode(true);
    const id = `passage_${++passageId}`;
    row.dataset.id = id;
    const input = row.querySelector(".passage-text");
    input.id = id;
    input.value = text;
    row.querySelector(".passage-label").htmlFor = id;
    const sourceInput = row.querySelector(".passage-source");
    sourceInput.id = `${id}_source`;
    sourceInput.value = source;
    row.querySelector(".source-label").htmlFor = sourceInput.id;
    row.querySelector(".remove-passage").addEventListener("click", () => {
      row.remove();
      resetResults();
      updatePassages();
      evidenceFields.lastElementChild.querySelector(".passage-text").focus();
    });
    evidenceFields.append(row);
    updatePassages();
    if (focus) input.focus();
  }

  function resetResults() {
    $("results").hidden = true;
    $("empty-state").hidden = false;
    $("error").hidden = true;
    $("status").textContent = "";
    $("result-state").textContent = "Awaiting your answer";
  }

  function resetForm() {
    form.reset();
    evidenceFields.replaceChildren();
    passageId = 0;
    addPassage();
    updateMode();
  }

  function passageName(evidence) {
    const index = [...evidenceFields.children].findIndex((row) => row.dataset.id === evidence.id);
    return evidence.source || `Passage ${index + 1}`;
  }

  function showEvidence(evidence, index) {
    $("source-name").textContent = passageName(evidence);
    $("source-text").textContent = evidence.text;
    $("evidence-explanation").textContent = evidence.explanation;
    const scores = [
      ["Weighted verification score", evidence.combined_score.toFixed(3)],
      ["Entity and number match", evidence.ner_score.toFixed(3)],
      ["Semantic similarity", evidence.embedding_score.toFixed(3)],
      ["NLI entailment", `${(evidence.nli_probabilities.entailment * 100).toFixed(1)}%`],
      ["NLI contradiction", `${(evidence.nli_probabilities.contradiction * 100).toFixed(1)}%`],
      ["NLI neutral", `${(evidence.nli_probabilities.neutral * 100).toFixed(1)}%`],
    ];
    $("signals").replaceChildren();
    for (const [name, value] of scores) {
      const term = document.createElement("dt");
      const definition = document.createElement("dd");
      term.textContent = name;
      definition.textContent = value;
      $("signals").append(term, definition);
    }
    document.querySelectorAll(".evidence-option").forEach((button, buttonIndex) => {
      button.setAttribute("aria-pressed", String(buttonIndex === index));
    });
  }

  function showClaim(claim, index) {
    const verdict = labels[claim.verdict];
    $("claim-heading").textContent = `Claim ${index + 1}`;
    $("claim-verdict").textContent = verdict.text;
    $("claim-verdict").className = `verdict-tag ${verdict.className}`;
    $("claim-text").textContent = claim.text;
    $("claim-explanation").textContent = claim.search
      ? `${claim.explanation} Searched ${claim.search.passages_searched} knowledge-base passages; the most relevant are shown below.`
      : claim.explanation;
    $("extraction-note").hidden = !claim.extraction_note;
    $("extraction-note").textContent = claim.extraction_note || "";
    $("extraction-note").className = `extraction-note${claim.review_required ? " needs-review" : ""}`;
    $("evidence-options").replaceChildren();
    claim.evidence.forEach((evidence, evidenceIndex) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "evidence-option";
      const source = document.createElement("span");
      source.textContent = passageName(evidence);
      const status = document.createElement("span");
      status.textContent = labels[evidence.verdict].text;
      status.className = `verdict-tag ${labels[evidence.verdict].className}`;
      button.append(source, status);
      button.addEventListener("click", () => showEvidence(evidence, evidenceIndex));
      $("evidence-options").append(button);
    });
    showEvidence(
      claim.evidence.find((item) => item.id === claim.verified_evidence_id),
      claim.evidence.findIndex((item) => item.id === claim.verified_evidence_id),
    );
    document.querySelectorAll(".claim").forEach((button, buttonIndex) => {
      button.setAttribute("aria-pressed", String(buttonIndex === index));
    });
  }

  function renderResults(result, answer) {
    const verdict = labels[result.verdict];
    $("overall").textContent = verdict.text;
    $("overall").className = verdict.className;
    const counts = Object.keys(labels).map((key) => {
      const count = result.claims.filter((claim) => claim.verdict === key).length;
      return count ? `${count} ${labels[key].text.toLowerCase()}` : null;
    }).filter(Boolean);
    const reviewCount = result.claims.filter((claim) => claim.review_required).length;
    if (reviewCount) counts.push(`${reviewCount} kept intact for review`);
    $("counts").textContent = counts.join(" / ");
    const annotated = $("annotated-answer");
    annotated.replaceChildren();
    // Python offsets count Unicode code points; JavaScript string slices count UTF-16 units.
    const characters = Array.from(answer);
    let position = 0;
    result.claims.forEach((claim, index) => {
      annotated.append(document.createTextNode(characters.slice(position, claim.start_char).join("")));
      const button = document.createElement("button");
      button.type = "button";
      button.className = `claim ${labels[claim.verdict].className}`;
      button.textContent = characters.slice(claim.start_char, claim.end_char).join("");
      button.setAttribute("aria-label", `Claim ${index + 1}: ${labels[claim.verdict].text}. ${claim.text}`);
      button.addEventListener("click", () => showClaim(claim, index));
      annotated.append(button);
      position = claim.end_char;
    });
    annotated.append(document.createTextNode(characters.slice(position).join("")));
    $("results").hidden = false;
    $("empty-state").hidden = true;
    $("result-state").textContent = `${result.claims.length} claim${result.claims.length === 1 ? "" : "s"} checked`;
    showClaim(result.claims[0], 0);
  }

  function errorMessage(data) {
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail)) {
      return data.detail.map((item) => {
        const location = item.loc.slice(1);
        const field = location[0] === "evidence"
          ? `Passage ${Number(location[1]) + 1}`
          : { question: "Question", answer: "AI answer" }[location[0]] || "Input";
        return `${field}: ${item.msg.replace(/^Value error, /, "")}`;
      }).join(". ");
    }
    return "Verification could not finish. Try again, or check the local server if this continues.";
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const answer = $("answer").value.trim();
    const body = { question: $("question").value.trim(), answer };
    if (!searching()) {
      body.evidence = [...evidenceFields.children].map((row) => ({
        id: row.dataset.id,
        text: row.querySelector(".passage-text").value.trim(),
        source: row.querySelector(".passage-source").value.trim() || null,
      }));
    }
    resetResults();
    $("empty-state").hidden = true;
    $("inputs").disabled = true;
    $("load-example").disabled = true;
    $("clear").disabled = true;
    pane.setAttribute("aria-busy", "true");
    $("verify").firstChild.textContent = "Verifying answer";
    $("result-state").textContent = "Verification in progress";
    const started = Date.now();
    $("status").textContent = searching()
      ? "Searching the visa knowledge base locally. The first run may take longer while models load."
      : "Checking your evidence locally. The first run may take longer while models load.";
    const loadingText = () => {
      const elapsed = Math.floor((Date.now() - started) / 1000);
      $("result-state").textContent = `Checking evidence (${elapsed}s)`;
    };
    loadingText();
    const timer = setInterval(loadingText, 1000);
    try {
      const response = await fetch(searching() ? "/api/v1/verify-search" : "/api/v1/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(response.status === 422 ? errorMessage(data) : errorMessage({}));
      }
      renderResults(await response.json(), answer);
      $("status").textContent = "Verification complete. Select a claim to compare its evidence.";
    } catch (error) {
      $("error").textContent = error instanceof TypeError
        ? "Cannot reach HalluciGuard. Check that the local server is running, then try again."
        : error.message;
      $("error").hidden = false;
      $("status").textContent = "";
      $("result-state").textContent = "Verification did not finish";
    } finally {
      clearInterval(timer);
      $("inputs").disabled = false;
      $("own-evidence").disabled = searching();
      $("load-example").disabled = false;
      $("clear").disabled = false;
      $("verify").firstChild.textContent = "Verify answer";
      pane.setAttribute("aria-busy", "false");
      updatePassages();
      if (window.matchMedia("(max-width: 720px)").matches) {
        pane.scrollIntoView({ behavior: "instant", block: "start" });
      }
    }
  });

  form.addEventListener("input", resetResults);
  $("add-evidence").addEventListener("click", () => {
    resetResults();
    addPassage("", "", true);
  });
  $("clear").addEventListener("click", () => { resetForm(); $("question").focus(); });
  $("load-example").addEventListener("click", () => {
    const mode = form.elements.mode.value;
    resetForm();
    form.elements.mode.value = mode;
    updateMode();
    $("question").value = "Can I work while studying in Australia on a student visa?";
    $("answer").value = "Student visa holders can work up to 60 hours per fortnight while their course is in session. They must have adequate health insurance for their whole stay.";
    if (!searching()) {
      evidenceFields.replaceChildren();
      addPassage("With this visa you can work up to 48 hours a fortnight when your course of study or training is in session.", "Department of Home Affairs, Student visa (subclass 500)");
      addPassage("You and your family members must have and maintain adequate health insurance for the whole of your stay in Australia.", "Department of Home Affairs, Student visa (subclass 500)");
    }
    $("status").textContent = "Example loaded: one claim gets the work-hours limit wrong. Verify to see the actual model results.";
    $("question").focus();
  });
  form.querySelectorAll('input[name="mode"]').forEach((input) => input.addEventListener("change", updateMode));
  addPassage();
  updateMode();
})();
