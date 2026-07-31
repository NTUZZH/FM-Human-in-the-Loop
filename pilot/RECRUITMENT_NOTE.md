# Recruitment note

Paste the body below into an email. Replace the bracketed parts. Attach
`y3_w8_pilot.html`. Do not attach the manifest: it contains the answer key.

Send one email per person and assign each person a code (R1, R2, R3, ...) in the
line where it appears, so the returned sheets can be told apart without a name.

---

**Subject:** 40 minutes of your judgement on work-order priorities (research, anonymous)

Dear [name],

I am [Ziheng Zhang, Research Fellow in the Infocomm Technology Cluster at the
Singapore Institute of Technology]. I am working on software that helps decide
which maintenance work order a team should start next, and I would value 40
minutes of your judgement.

**What the task is.** You open one file in your browser and see 56 pairs of real
maintenance work orders, drawn from a public research dataset of North American
university campuses. Each pair shows two jobs sitting in the same site's queue at
the same moment, with the description as it was written, the system and component,
the recorded priority, how long each job has been waiting, and an estimate of the
labour involved. For each pair you say which of the two you would start first, how
confident you are, and, if you want, one line on why. There are no right answers.
The whole point is to find out how experienced practitioners actually weigh these
against each other.

**Why I am asking you.** The software has to learn from a supervisor's judgement,
and at the moment I only have a simulated supervisor. I need to know whether real
practitioners agree with each other about which job is more urgent, and whether
that judgement tracks anything visible in the record. Your answers are the only
way to find out.

**What it costs you.** About 40 minutes, at your own desk, whenever suits. You can
stop halfway and come back as long as you leave the browser tab open. Nothing is
installed, nothing is uploaded, and the file works offline.

**What happens to your answers.** Press one button at the end and your browser
saves a small CSV file; email it back to me. Please use the participant code
**[R1]** when the file asks for one. I record no name, no employer, and nothing
that identifies you. Answers are reported only as group statistics, for example
how often practitioners agree with each other, in a research paper on automating
work-order dispatch. Individual answers are never shown or quoted. You may stop at
any time, and if you would rather your partial answers were discarded, say so and
they will be.

**If you are willing**, just open the attached file and start. If anything is
unclear, or if the file does not open properly, reply and I will sort it out.

With thanks,

[Ziheng Zhang]
[ziheng.zhang@singaporetech.edu.sg]

---

## Practical notes for the sender (not part of the email)

- **Target 5 participants, minimum 3, maximum 8.** More than 8 adds little, since
  the pairs, not the raters, are the statistical sample.
- **Anyone who dispatches or plans maintenance work qualifies**: FM supervisors,
  planners, senior technicians, CMMS coordinators. Record each person's role and
  years of experience from the form, nothing more.
- **Assign codes yourself and keep the code-to-person mapping out of the repository.**
  The analysis never needs it.
- If a participant cannot run the HTML file, send `y3_w8_response_template.csv`
  instead. The items are in the same order in both, and the analysis reads either.
- Drop returned CSVs into `pilot/responses/` and run
  `python scripts/y3_w8_pilot_analyse.py`. Nothing else is needed.
