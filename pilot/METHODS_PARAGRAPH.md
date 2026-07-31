# Manuscript text for the W8 pilot

Everything below is drop-in LaTeX for `paper/`. Only the numbers need filling,
and they are all macros, so the manuscript's "no literal numbers" rule holds and
an unfilled value renders as the visible `\TBD` marker.

**Do not paste anything until the responses are in and
`scripts/y3_w8_pilot_analyse.py` has run.** The numbers come from
`results/y3_w8/pilot_analysis.json`; the mapping from macro to JSON key is in the
table at the end of this file.

---

## 1. Where it goes

**Section "Results", as a new subsection placed between "The boundary" and
"Transfer to held-out campuses"**, with the heading *Practitioner grounding of the
supervisor premises*.

That block of the Results section already answers "how far does this carry", first
across queue regimes and then across campuses; "and does it hold with real people"
belongs with them. Putting it earlier would interrupt the argument from dispatch
quality to recovery to ablation, and putting it last would let a five-person pilot
close the Results section, which over-weights it.

Two other edits go with it:

- one forward sentence in *The private-information supervisor* (Section
  `sec:supervisor`), so a reader meets the premises and their evidence together;
- rewrites of the first two paragraphs of *Limitations and future work* (Section
  `sec:disc-limits`), which currently name this study as work that would be done.
  Both drafts are in section 5 below.

**If the editor asks for length back**, move the subsection wholesale to the
Supplemental Materials, keep the boundary sentence and the two headline numbers
inside the limitations paragraph, and cross-reference the supplement by name. The
subsection is written so that it survives that move without rewriting.

## 2. What the pilot licenses in that slot, and what it does not

It licenses exactly one claim: **the two premises the supervisor model is built
on hold for practising facility staff judging real orders from the same corpus.**
Concretely, that an urgency ordering exists which practitioners share beyond the
recorded priority class, and that part of that ordering is a function of the
observable order attributes the estimator reads.

It licenses nothing about the loop. The boundary sentence in section 4 states
that, and it must travel with the numbers wherever they are quoted.

## 3. The subsection

```latex
\subsection{Practitioner grounding of the supervisor premises}\label{sec:res-pilot}
The supervisor of Section~\ref{sec:supervisor} rests on two premises: that an
urgency ordering exists which practitioners share beyond the recorded class, and
that part of it is a function of observable order attributes. Both were tested
with practising facility staff on real orders. \pilotNrater{} practitioners each
judged \pilotNpairs{} pairs of FMUCD work orders drawn from the six campuses this
study uses, sampled by a fixed seeded rule and stratified so that some pairs carry
equal recorded classes, some differ by one class, and some set the recorded class
against what the observable attributes suggest. Each pair showed only what a
supervisor holds at the moment of dispatch, namely the written description, the
system and component, the recorded class with its target response time, how long
the order had waited, and the estimated labour; realised duration, cost, and
close-out date were withheld, because they are outcomes rather than inputs. The
hypotheses and the statistics were registered before any response was collected.

Practitioners agreed with one another well above chance
(Krippendorff's $\alpha = \pilotAlpha{}$, 95\% confidence interval
[\pilotAlphaCIlo{}, \pilotAlphaCIhi{}], permutation $p = \pilotAlphaP{}$), and
they still agreed on the \pilotNSone{} pairs whose recorded classes are equal
($\alpha = \pilotAlphaSone{}$), where the record itself offers no guidance. On the
\pilotClassAgreeN{} pairs whose recorded classes differ, the practitioners'
majority followed the record in only \pilotClassAgree{} of them (95\% confidence
interval [\pilotClassAgreeCIlo{}, \pilotClassAgreeCIhi{}]), which measures
directly what Section~\ref{sec:supervisor} assumes: the recorded class carries
real information, and it does not settle which job is served first. Their majority
judgement was predictable out of sample from the three observable attributes the
estimator reads, reaching an area under the ROC curve of \pilotAUC{} (95\%
confidence interval [\pilotAUCCIlo{}, \pilotAUCCIhi{}]) against \pilotAUCclass{}
for the recorded class alone, which is the real-data analogue of $\beta > 0$.
Read through the latent of Section~\ref{sec:supervisor}, that predictability
corresponds to a recoverable share between \pilotBetaLo{} and \pilotBetaHi{}, and
the observed disagreement between practitioners to an override noise level between
\pilotEpsLo{} and \pilotEpsHi{}. Both are translations under assumptions set out
with the instrument in the Supplemental Materials, not measurements of the model's
parameters, and both land inside the ranges this study sweeps.

The pilot asks practitioners to rank pairs of real work orders drawn from the same
corpus the study models; it does not place a practitioner inside the dispatch
loop. It therefore grounds the two premises just stated and grounds nothing else.
It does not validate the correction loop, the dispatching results, or any reported
reduction in true weighted tardiness, every one of which is measured against the
simulated supervisor. The instrument, the pre-registration, and the analysis are
released with the code.
```

**If the instrument check did not pass** (the analysis prints
`gate NOT PASSED`), append this sentence to the second paragraph, before the
boundary paragraph:

```latex
Within-rater consistency, measured from \pilotNrepeat{} pairs shown twice in
swapped positions, was \pilotConsist{} (95\% confidence interval
[\pilotConsistCIlo{}, \pilotConsistCIhi{}]), which at this number of repeats does
not separate a stable judgement from an unstable one, so the agreement and
predictability figures above should be read as a lower bound on what a
better-powered instrument would find.
```

## 4. The boundary sentence, on its own

Quote this verbatim wherever the pilot's numbers appear outside
Section~\ref{sec:res-pilot}, including any response to reviewers:

> The pilot asks practitioners to rank pairs of real work orders drawn from the
> same corpus the study models; it does not place a practitioner inside the
> dispatch loop. It therefore grounds two premises of the supervisor model, that
> an urgency ordering exists which practitioners share beyond the recorded
> priority class and that it is partly predictable from the observable order
> attributes the estimator reads, and it grounds nothing else. It does not
> validate the correction loop, the dispatching results, or any reported
> reduction in true weighted tardiness, every one of which is measured against a
> simulated supervisor.

## 5. The two edits that go with it

**(a) In `sec:supervisor`**, after the sentence ending "...the noise $z_j$ is
redrawn per order.", add:

```latex
Both premises behind this construction, that practitioners share an urgency
ordering the recorded class does not capture and that part of it follows the
observable features, are tested against practitioner judgements on real orders in
Section~\ref{sec:res-pilot}.
```

**(b) In `sec:disc-limits`**, the first two paragraphs currently promise this
study as future work. Replace the closing sentence of the first paragraph
("A pilot with real practitioners would test whether...") with:

```latex
A pilot with practising facility staff (Section~\ref{sec:res-pilot}) confirms that
the premises the simulated supervisor is built on hold for real people judging
real orders, but it measures judgement rather than intervention: whether a
supervisor's override pattern in a live loop resembles the simulated one remains
open, and logging the supervisor's intended pick beside the executed one is the
change that would settle it, one the override-noise ablation already prices, since
intent labels would hold the reduction over the rule at \EpsPreferredUB{} under
the highest noise level tested (Section~\ref{sec:res-ablation}).
```

and replace the closing sentence of the second paragraph ("A field study
recording a supervisor's own urgency assessment beside the recorded class would
replace the proxy with a direct measurement.") with:

```latex
The practitioner pilot of Section~\ref{sec:res-pilot} replaces this proxy with a
direct judgement on the same corpus and reaches the same conclusion from a
different direction; a field study logging a supervisor's urgency assessment
beside the recorded class, on that supervisor's own portfolio, is what would turn
both readings into a site-specific measurement.
```

## 6. Macro block

Paste into `paper/macros.tex`, in the results-macro block, with the values from
`results/y3_w8/pilot_analysis.json`. Every one is `\TBD` until the pilot runs.

```latex
% ---- W8 practitioner pilot (results/y3_w8/pilot_analysis.json) ------------ %
\newcommand{\pilotNrater}{\TBD}        % n_raters
\newcommand{\pilotNpairs}{\TBD}        % design.n_unique_pairs
\newcommand{\pilotNrepeat}{\TBD}       % design.n_repeat_pairs
\newcommand{\pilotNSone}{\TBD}         % design.strata.S1_equal_class
\newcommand{\pilotAlpha}{\TBD}         % agreement.krippendorff_alpha
\newcommand{\pilotAlphaCIlo}{\TBD}     % agreement.ci95[0]
\newcommand{\pilotAlphaCIhi}{\TBD}     % agreement.ci95[1]
\newcommand{\pilotAlphaP}{\TBD}        % agreement.permutation_p; a permutation
                                       % p is bounded below by 1/(B+1), so write
                                       % "<0.001" rather than the literal 0.0001
\newcommand{\pilotAlphaSone}{\TBD}     % agreement.per_stratum.S1_equal_class.krippendorff_alpha
\newcommand{\pilotClassAgreeN}{\TBD}   % class_test.n_scored
\newcommand{\pilotClassAgree}{\TBD}    % class_test.rate, as a percentage
\newcommand{\pilotClassAgreeCIlo}{\TBD}% class_test.ci95_wilson[0]
\newcommand{\pilotClassAgreeCIhi}{\TBD}% class_test.ci95_wilson[1]
\newcommand{\pilotAUC}{\TBD}           % predictability.models.M2_attributes.auc
\newcommand{\pilotAUCCIlo}{\TBD}       % predictability.models.M2_attributes.auc_ci95[0]
\newcommand{\pilotAUCCIhi}{\TBD}       % predictability.models.M2_attributes.auc_ci95[1]
\newcommand{\pilotAUCclass}{\TBD}      % predictability.models.M1_recorded_class.auc
\newcommand{\pilotConsist}{\TBD}       % within_rater.pooled_rate
\newcommand{\pilotConsistCIlo}{\TBD}   % within_rater.ci95_exact[0]
\newcommand{\pilotConsistCIhi}{\TBD}   % within_rater.ci95_exact[1]
\newcommand{\pilotBetaLo}{\TBD}        % translation.beta.implied_beta_range[0]
\newcommand{\pilotBetaHi}{\TBD}        % translation.beta.implied_beta_range[1]
\newcommand{\pilotEpsLo}{\TBD}         % translation.epsilon.implied_epsilon_range[0]
\newcommand{\pilotEpsHi}{\TBD}         % translation.epsilon.implied_epsilon_range[1]
```

## 7. If a claim in the paragraph is not supported

The paragraph above is written for the outcome the design expects. The
pre-registration fixes what to write in every other case; the short version:

- **Agreement at or near chance.** Delete the predictability and translation
  sentences, report the agreement figure with its interval, and state that the
  pilot did not find a shared urgency ordering at this sample size, so the
  premise remains an assumption. Keep the number in the paper.
- **Agreement holds but the attributes do not predict it.** Keep the first half,
  replace the predictability sentence with the measured area and its interval,
  and state that practitioners agree on something the three coded attributes do
  not capture, which points at the free text rather than against the premise.
- **The majority follows the recorded class almost always.** Report it. It says
  the class is a better record than the study assumes on these campuses, which
  bounds the headroom the layer has there, and it belongs beside the Payoff
  Condition of Section~\ref{sec:payoff}.

Every measured number goes into the paper whichever way it falls. Emphasis and
order may change; a number may not.
