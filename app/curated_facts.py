"""
Single source of truth for curated facts: high-value information that kept
losing the similarity-ranking race against larger, noisier pages, so it's
indexed as its own guaranteed-to-rank-well chunk instead.

IMPORTANT: this file is imported by BOTH ingest/build_index.py (to build
the index) and app/rag.py (to force-include facts by keyword trigger).
Each fact's position in this list determines its chunk id ("curated-fact-N")
— that id is used consistently by both files because they both read from
THIS list, so content and triggers can never drift out of sync the way
they did before this file existed. When adding a new fact, just append a
dict here with "text", "url", "title", and "trigger" (a regex pattern
string) — nothing else needs to change in either build_index.py or rag.py.
"""

CURATED_FACTS = [
    {
        "text": "The Lahore Leads University Islamabad Campus is located "
                "at Service Road (South), Near Metro Station G-13, "
                "Srinagar Highway, Islamabad — also described as G-12, "
                "opposite the G-13 Metro Bus stop.",
        "url": "https://leads.edu.pk/islamabad-campus/",
        "title": "Islamabad Campus — Address",
        "trigger": r"\b(address|location|located|where)\b",
    },
    {
        "text": "The Islamabad campus contact number, including WhatsApp, "
                "is +92 314 4477774. You can also email "
                "admissions@leads.edu.pk for admissions queries.",
        "url": "https://leads.edu.pk/%f0%9f%8e%93-admissions-now-open-for-fall-2026-intake/",
        "title": "Islamabad Campus — Phone/WhatsApp",
        "trigger": r"\b(phone|whatsapp|contact\s+number|call)\b",
    },
    {
        "text": "The Campus Director of Lahore Leads University, "
                "Islamabad Campus, is Professor Dr Munawar Iqbal Ahmed.",
        "url": "https://leads.edu.pk/islamabad-campus/",
        "title": "Islamabad Campus — Campus Director",
        "trigger": r"\bdirector\b",
    },
    {
        "text": "The Lahore Leads University Islamabad Campus has "
                "officially received HEC (Higher Education Commission) "
                "NOC approval.",
        "url": "https://leads.edu.pk/congratulations-leads-university-islamabad-campus-is-now-officially-hec-noc-approved/",
        "title": "Islamabad Campus — HEC Approval",
        "trigger": r"\b(hec|noc|approv(al|ed))\b",
    },
    {
        # CORRECTED against the official printed admissions brochure
        # (undergraduate programs flyer) — the Islamabad campus offers
        # far fewer standalone degrees than an earlier version of this
        # fact claimed. AI/Data Sciences/Cyber Security/Software
        # Engineering and Digital Marketing/Data Analytics/Finance/
        # Banking are SPECIALIZATION TRACKS within BSCS and BBA
        # respectively — NOT separate standalone degree programs. There
        # is currently NO Engineering program, NO MPhil, and NO separate
        # BS Software Engineering / BS Data Science / BS Cybersecurity /
        # BS Fintech / BS Accounting & Finance / MBA / ADP Accounting & 
        # Finance / ADP Fintech at the Islamabad campus. Do not list any
        # of those — this fact supersedes any broader university-wide
        # catalog mentioned elsewhere in retrieved content.
        "text": "At the Islamabad campus, the following programs are "
                "currently offered:\n\n"
                "Undergraduate (BS):\n"
                "1) Bachelor in Computer Sciences (BSCS) — with "
                "specializations in AI, Data Sciences, Cyber Security, "
                "and Software Engineering\n"
                "2) Bachelor in Business Administration (BBA) — with "
                "specializations in Digital Marketing, Data Analytics, "
                "Finance, and Banking\n\n"
                "Associate Degree Program (2 years):\n"
                "1) ADP in Computer Sciences\n"
                "2) ADP in Business Management\n\n"
                "Additionally, short/prep courses such as IELTS, CA, and "
                "ACCA may also be offered.",
        "url": "https://leads.edu.pk/islamabad-campus/",
        "title": "Islamabad Campus — Programs Offered (Corrected, per official brochure)",
        "trigger": r"\b(programs?\s+offer|what\s+programs|courses?\s+offer|engineering|medical|pharmacy|nursing|architecture|law\b)\b",
    },
    {
        "text": "The official Islamabad Campus website is "
                "https://isb.leads.edu.pk. To apply for admission, use "
                "the direct application link: "
                "https://apply.leads.edu.pk/registration/iao",
        "url": "https://isb.leads.edu.pk",
        "title": "Islamabad Campus — Website & Admission Link",
        "trigger": r"\b(website|apply|admission\s+link|portal|register|registration)\b",
    },
    {
        "text": "A minimum of 50 percent marks in FSC is required for admission "
                "to BBA, BS Computer Science, and ADP at the Islamabad campus.",
        "url": "https://isb.leads.edu.pk",
        "title": "Islamabad Campus — Admission Marks Requirement",
        "trigger": r"\b(minimum\s+marks|marks\s+requirement|50\s*%\s+marks)\b",
    },
    {
        # Corrected to include Software Engineering, which the official
        # brochure lists as a 4th BSCS specialization track (previously
        # missing from this fact).
        "text": "BS Computer Science (BSCS) at the Islamabad campus "
                "offers specializations in AI, Data Sciences, Cyber "
                "Security, and Software Engineering. BBA at the "
                "Islamabad campus offers specializations in Digital "
                "Marketing, Data Analytics, Finance, and Banking.",
        "url": "https://leads.edu.pk/islamabad-campus/",
        "title": "Islamabad Campus — Program Specializations",
        "trigger": r"specializ|specialis|cializ|cialis",
    },
    {
        "text": "Yes, hostel and transport facilities are available at "
                "the Islamabad campus.",
        "url": "https://leads.edu.pk/islamabad-campus/",
        "title": "Islamabad Campus — Hostel & Transport",
        "trigger": r"\b(hostel|transport)\b",
    },
    {
        "text": "Boys Hostel (Munawar Boys Hostel, M.B.H): Contact "
                "number 0333-9564736. Address: Old Kashmir Highway, "
                "G-12/1, Iqbal Town, Islamabad. It's within walking "
                "distance of the Leads Islamabad campus.",
        "url": "https://leads.edu.pk/islamabad-campus/",
        "title": "Islamabad Campus — Boys Hostel Details",
        "trigger": r"\b(hostel|boys?\s+hostel|male\s+hostel)\b",
    },
    {
        "text": "Girls Hostel (Munawar Girls Hostel, M.G.H): Contact "
                "numbers 0313-9564736, 0303-8190786, or 0333-5751808. "
                "Address: Near G-13 Metro Station, SLS School Street "
                "No. 2-B, Service Road G-12/1, Islamabad. It's within "
                "walking distance of the Leads Islamabad campus. It's a "
                "separate, dedicated building with 24/7 security guards "
                "and CCTV surveillance, operating since 2016. Rooms are "
                "fully furnished (bi-seater, tri-seater, and "
                "tetra-seater options), with laundry facilities, "
                "nutritious food options, a TV lounge, and study areas. "
                "It's also about 5 minutes from the G-13 Metro/NUST Bus "
                "Stop, with FAST University Islamabad (H-11/4) and IIUI "
                "Girls Campus accessible by university transport or "
                "metro.",
        "url": "https://leads.edu.pk/islamabad-campus/",
        "title": "Islamabad Campus — Girls Hostel Details",
        "trigger": r"\b(hostel|girls?\s+hostel|female\s+hostel)\b",
    },
    {
        # Derived, not invented: the confirmed 1st-semester fee tables
        # explicitly label Admission Fee, Enrollment Fee, and Library Fee
        # as "(once)" and Tuition Fee / Examination Fee as "(per
        # semester)". The 2nd semester total is therefore just the
        # recurring items, dropping the one-time ones — a direct
        # consequence of data already trusted elsewhere, not a guess.
        "text": "2nd semester fees at the Islamabad campus (the one-time "
                "Admission, Enrollment, and Library fees are only charged "
                "in the 1st semester — 2nd semester only has the "
                "recurring per-semester charges): "
                "BS Computer Science (BSCS): Tuition Fee 120,000 PKR + "
                "Examination Fee 5,000 PKR = 125,000 PKR total for the "
                "2nd semester. "
                "BBA: Tuition Fee 100,000 PKR + Examination Fee 5,000 PKR "
                "= 105,000 PKR total for the 2nd semester. "
                "ADP (Business Administration / Computer Science): "
                "Tuition Fee 65,000 PKR + Examination Fee 5,000 PKR = "
                "70,000 PKR total for the 2nd semester.",
        "url": "https://leads.edu.pk/islamabad-campus/",
        "title": "Islamabad Campus — 2nd Semester Fee Structure",
        "trigger": r"\b(2nd\s+semester|second\s+semester)\b",
    },
    {
        "text": "Scholarship eligibility is based on a student's aggregate. "
                "There are two cases: (1) If a student has NOT yet "
                "received their Intermediate SECOND YEAR result (common "
                "for recently-examined students still awaiting results), "
                "aggregate is calculated PROVISIONALLY as Matric % + "
                "Intermediate FIRST YEAR % only. (2) If a student HAS "
                "already received their complete Intermediate (FSc/ICS) "
                "result — both years — aggregate is calculated as Matric "
                "% + the COMPLETE Intermediate percentage (both years "
                "combined), NOT just the first year. Always ask a student "
                "whether they have their full/complete Intermediate "
                "result yet before deciding which calculation applies.",
        "url": "https://leads.edu.pk/islamabad-campus/",
        "title": "Islamabad Campus — Aggregate Calculation Formula",
        "trigger": r"\b(aggregate|merit\s+scholarship)\b",
    },
    {
        "text": "Merit scholarship: if a student's aggregate is 85% or "
                "above, they are offered a 75% discount on the tuition "
                "fee. (See the Aggregate Calculation Formula fact for how "
                "the aggregate itself is calculated, which depends on "
                "whether the student has their complete Intermediate "
                "result yet.)",
        "url": "https://leads.edu.pk/islamabad-campus/",
        "title": "Islamabad Campus — Merit Scholarship (85%+ Aggregate = 75% Tuition Discount)",
        "trigger": r"\b(aggregate|merit\s+scholarship|85\s*%)\b|\bfee\b|\btuition\b|schola|scolar|\bdiscount\b",
    },
    {
        "text": "Need/circumstance-based discount: students who belong to "
                "a needy financial background, or who apply for early "
                "admission, are offered a 25% waiver on the tuition fee. "
                "The early-admission portion of this offer is "
                "time-limited to applications submitted before 25th "
                "August (for the current intake). NOTE: this 25% tier is "
                "for needy background / early admission only — orphan "
                "students get a DIFFERENT, higher discount (see the "
                "separate Orphan Discount fact).",
        "url": "https://leads.edu.pk/islamabad-campus/",
        "title": "Islamabad Campus — Need-Based / Early-Admission Discount (25% Tuition)",
        "trigger": r"\b(needy|early\s+admission|financial\s+need|25\s*%)\b|\bfee\b|\btuition\b|schola|scolar|\bdiscount\b",
    },
    {
        "text": "Orphan students are offered a 50% discount on the "
                "tuition fee. This is a HIGHER discount than the general "
                "need-based tier (25%) — orphan status specifically gets "
                "50%, not 25%.",
        "url": "https://leads.edu.pk/islamabad-campus/",
        "title": "Islamabad Campus — Orphan Discount (50% Tuition)",
        "trigger": r"\borphan\b|\bfee\b|\btuition\b|schola|scolar|\bdiscount\b",
    },
    {
        "text": "Students from FATA (Federally Administered Tribal Areas) "
                "and other Balochistan cities can avail a 50% discount on "
                "fees based on financial need.",
        "url": "https://leads.edu.pk/islamabad-campus/",
        "title": "Islamabad Campus — FATA & Balochistan Need-Based Discount (50%)",
        "trigger": r"\b(fata|balochistan|baloch)\b|\bfee\b|\btuition\b|schola|scolar|\bdiscount\b",
    },
    {
        "text": "If a student's Matric aggregate component is 50% or "
                "above but their PROVISIONAL aggregate (using Intermediate "
                "FIRST YEAR only) looks low because their first year marks "
                "were below 50% (for example 47%), first check: has the "
                "student already received their COMPLETE Intermediate "
                "(both years) result? If yes, recalculate their aggregate "
                "using Matric % + the COMPLETE Intermediate % instead of "
                "first-year-only — this may put them in a completely "
                "different, better eligibility bracket. If they've "
                "appeared in second-year exams but do NOT have the result "
                "yet, do not tell them they are ineligible — advise them "
                "to submit their documents and apply so the admissions "
                "team can reassess once the complete result is available.",
        "url": "https://leads.edu.pk/islamabad-campus/",
        "title": "Islamabad Campus — Below-50% First-Year, Second-Year-Appeared Case",
        "trigger": r"\b(aggregate|first\s+year|1st\s+year|second\s+year|eligib)\b",
    },
    {
        "text": "If a student's original documents (such as the Matric "
                "or Intermediate marksheet, or any other required "
                "document) are missing, a photocopy of that document can "
                "also be accepted for the admission process.",
        "url": "https://leads.edu.pk/islamabad-campus/",
        "title": "Islamabad Campus — Missing Original Documents Policy",
        "trigger": r"\b(document|documents|marksheet|original|photocopy|missing|copy)\b",
    },
    {
        "text": "Students from a pre-medical background are also eligible "
                "for admission into Computer Science programs (BS "
                "Computer Science, ADP Computer Science) at the Islamabad "
                "campus. They follow the exact same admission process as "
                "all other students — no separate or additional "
                "requirement applies just because their background is "
                "pre-medical rather than pre-engineering/computer science.",
        "url": "https://leads.edu.pk/islamabad-campus/",
        "title": "Islamabad Campus — Pre-Medical Students Eligible for Computer Science",
        "trigger": r"\bpre-?medical\b",
    },
    {
        "text": "Currently, no admission test is required to get "
                "admission at the Islamabad campus. Admission decisions "
                "are based solely on the student's Matric and "
                "Intermediate aggregate (see the Aggregate Calculation "
                "Formula fact for how that's calculated) — there is no "
                "separate entry/admission test for undergraduate programs "
                "like BS Computer Science, ADP, or BBA right now. "
                "HOWEVER: this is tied to Early Bird admission — seats "
                "with guaranteed scholarships are limited and are being "
                "filled on a rolling basis. Once those Early Bird seats "
                "are filled, future applicants may be required to appear "
                "for a mandatory entrance test instead, with admission "
                "decided by the test merit list. Students should be "
                "encouraged to apply promptly to secure a no-test Early "
                "Bird seat with a guaranteed scholarship while it's still "
                "available.",
        "url": "https://leads.edu.pk/islamabad-campus/",
        "title": "Islamabad Campus — No Admission Test Required (Currently, Early Bird)",
        "trigger": r"\b(admission\s+test|entry\s+test|entrance\s+test)\b",
    },
    {
        "text": "Early Bird admission seats with guaranteed scholarships "
                "are limited and are being filled on a rolling basis. "
                "Students are encouraged to apply as soon as possible to "
                "secure an Early Bird seat and lock in a guaranteed "
                "scholarship without needing to take an entrance test. "
                "Once Early Bird seats are filled, subsequent applicants "
                "may be required to appear for a mandatory entrance test, "
                "with admission decided based on the test merit list "
                "instead of the usual aggregate-based process.",
        "url": "https://leads.edu.pk/islamabad-campus/",
        "title": "Islamabad Campus — Early Bird Seats Limited, Apply Promptly",
        "trigger": r"\b(early\s+bird|seats?\s+fill|hurry|apply\s+(soon|quickly|now)|guaranteed\s+scholarship)\b",
    },
    {
        "text": "Lahore Leads University Islamabad Campus combines modern "
                "interactive education with complete cultural respect: "
                "male and female students share the same academic "
                "programs and classes, preparing them for professional, "
                "real-world environments. Classrooms feature organized, "
                "separate row arrangements for male and female students "
                "to maintain focus, modesty, and mutual respect.",
        "url": "https://leads.edu.pk/islamabad-campus/",
        "title": "Islamabad Campus — Co-Education & Classroom Arrangement",
        "trigger": r"\b(co-?education|coed|separate\s+class|gender\s+segregat|boys?\s+only|girls?\s+only|male\s+and\s+female|mixed\s+class|seating)\b",
    },
    {
        "text": "Buying a degree without genuinely completing the required "
                "coursework and attending as a real, enrolled student is "
                "NOT permitted — Lahore Leads University does not sell "
                "degrees. Similarly, obtaining a degree entirely from "
                "abroad without ever physically attending or enrolling "
                "as a proper student is NOT permitted — there is no "
                "remote-only or attendance-free degree option for "
                "students who wish to stay in their home country while "
                "avoiding actual study or enrollment. All students, "
                "including international/overseas applicants who "
                "genuinely want to study, must go through the standard "
                "admission process and actually attend/study.",
        "url": "https://leads.edu.pk/islamabad-campus/",
        "title": "Islamabad Campus — Academic Integrity (No Degree Buying / No Remote-Only Degrees)",
        "trigger": r"\b(buy\s+(a\s+)?degree|purchase\s+(a\s+)?degree|degree\s+without\s+(attending|studying|coming)|without\s+attending|sitting\s+(there|abroad|at\s+home)|from\s+abroad\s+without|no\s+need\s+to\s+attend)\b",
    },
    {
        # Genuinely Lahore-specific, NOT Islamabad's — tagged
        # "university-wide" (not the usual "islamabad" default) so the
        # campus-scoping rules treat it correctly. This is fine and SHOULD
        # be shared in two situations: (1) someone explicitly asks for
        # Lahore/main campus contact info, or (2) the bot is redirecting
        # someone to the main campus per the unavailable-field-or-program
        # rule (e.g. they asked about Engineering/Medical/etc., which
        # Islamabad doesn't offer). It should NOT be presented as if it
        # were Islamabad's own contact info in any other context.
        "text": "Lahore Main Campus contact details: Lahore Leads "
                "University, DHA Phase V, Kamahan Road, Lahore, "
                "Pakistan. Phone: 042-35927-411, 042-35927-413, "
                "042-35927-415, or 0304-1111-552. Email: "
                "admissions@leads.edu.pk.",
        "url": "https://leads.edu.pk/contact-us/",
        "title": "Lahore Main Campus — Contact Details",
        "campus": "university-wide",
        "trigger": r"\blahore\b|\bmain\s+campus\b|\bengineering\b|\bmedical\b|\bpharmacy\b|\bnursing\b|\barchitecture\b",
    },
    {
        "text": "To continue receiving any scholarship or fee "
                "discount/waiver at the Islamabad campus — whether it's "
                "the merit scholarship, need-based waiver, orphan "
                "discount, or the FATA/Balochistan discount — a student "
                "must maintain a minimum CGPA of 3.0. Falling below a "
                "3.0 CGPA may result in losing the scholarship.",
        "url": "https://leads.edu.pk/islamabad-campus/",
        "title": "Islamabad Campus — Scholarship Maintenance Requirement (3.0 CGPA)",
        "trigger": r"\bcgpa\b|\bgpa\b|maintain|keep\s+(my|the|a)\s+scholarship|minimum\s+grade",
    },
]
