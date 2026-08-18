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
        "text": "At the Islamabad campus, the following programs are currently offered:\n\n"
                "Department of Computer Science:\n"
                "- ADP Computer Science\n"
                "- ADP Software Engineering\n"
                "- BS Computer Science\n"
                "- BS Software Engineering\n"
                "- BS Data Science\n"
                "- BS Cybersecurity\n"
                "- MPhil Computer Science\n\n"
                "Department of Business Administration:\n"
                "- ADP Business Administration\n"
                "- ADP Accounting & Finance\n"
                "- ADP Fintech\n"
                "- ADP Business & Information System\n"
                "- BS Fintech\n"
                "- BS Accounting & Finance\n"
                "- BS Business & Information System\n"
                "- BBA\n"
                "- MBA\n\n"
                "Additionally, short/prep courses such as IELTS may also be offered.",
        "url": "https://leads.edu.pk/islamabad-campus/",
        "title": "Islamabad Campus — Programs Offered",
        "trigger": r"\b(programs?\s+offer|what\s+programs|courses?\s+offer)\b",
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
        "text": "BS Computer Science (BSCS) at the Islamabad campus offers "
                "specializations in AI, Cyber Security, and Data Science. "
                "BBA at the Islamabad campus offers specializations in "
                "Finance, Banking, Data Analytics, and Digital Marketing.",
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
        "text": "Scholarship eligibility is based on a student's aggregate, "
                "calculated as: Matric percentage + Intermediate (FSc/ICS/"
                "equivalent) FIRST YEAR percentage only. The Intermediate "
                "second year result is NOT included in this aggregate "
                "calculation.",
        "url": "https://leads.edu.pk/islamabad-campus/",
        "title": "Islamabad Campus — Aggregate Calculation Formula",
        "trigger": r"\b(aggregate|merit\s+scholarship)\b",
    },
    {
        "text": "Merit scholarship: if a student's aggregate (Matric % + "
                "Intermediate 1st year % only) is 85% or above, they are "
                "offered a 75% discount on the tuition fee.",
        "url": "https://leads.edu.pk/islamabad-campus/",
        "title": "Islamabad Campus — Merit Scholarship (85%+ Aggregate = 75% Tuition Discount)",
        "trigger": r"\b(aggregate|merit\s+scholarship|85\s*%)\b|\bfee\b|\btuition\b|\bscholarship\b|\bdiscount\b",
    },
    {
        "text": "Need/circumstance-based discount: students who belong to "
                "a needy financial background, are orphans, or apply for "
                "early admission are offered a 25% waiver on the tuition "
                "fee. The early-admission portion of this offer is "
                "time-limited to applications submitted before 25th "
                "August (for the current intake).",
        "url": "https://leads.edu.pk/islamabad-campus/",
        "title": "Islamabad Campus — Need/Orphan/Early-Admission Discount (25% Tuition)",
        "trigger": r"\b(needy|orphan|early\s+admission|financial\s+need|25\s*%)\b|\bfee\b|\btuition\b|\bscholarship\b|\bdiscount\b",
    },
    {
        "text": "Students from FATA (Federally Administered Tribal Areas) "
                "and other Balochistan cities can avail a 50% discount on "
                "fees based on financial need.",
        "url": "https://leads.edu.pk/islamabad-campus/",
        "title": "Islamabad Campus — FATA & Balochistan Need-Based Discount (50%)",
        "trigger": r"\b(fata|balochistan|baloch)\b|\bfee\b|\btuition\b|\bscholarship\b|\bdiscount\b",
    },
    {
        "text": "If a student's Matric aggregate component is 50% or "
                "above but their Intermediate FIRST YEAR marks are below "
                "50% (for example 47%), and the student has ALREADY "
                "appeared in their Intermediate SECOND YEAR exams, do not "
                "tell them they are ineligible or reject them. Instead, "
                "advise them to submit their documents and apply for "
                "admission so the admissions team can review their case "
                "directly.",
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
]
