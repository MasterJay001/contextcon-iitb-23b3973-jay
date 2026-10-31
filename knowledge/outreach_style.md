# Outreach style guide (used by the Outreach agent)

Purpose: a draft a recruiter can approve in ten seconds. Drafts are queued for approval; nothing is sent automatically, and during the event
sends go only to team allowlist addresses. No contact data about candidates is ever generated or used.

## Rules

1. 60-90 words. Subject under 8 words. One clear ask: a 20-minute conversation.
2. Open with ONE verified, specific fact from the candidate's own history (role title + company, or a domain they worked in). Use the exact words from the record.
3. Say why the opportunity matches that fact in one sentence. Do not flatter and do not invent achievements.
4. If a path in exists (an ex-colleague at the hiring company), mention the shared employer and years, using the placeholder {{COLLEAGUE_NAME}}. The renderer fills real names.
5. Address the person with the placeholder {{CANDIDATE_FIRST_NAME}}. The model never sees real names.
6. No salary claims, no urgency pressure, no mention of employment status, no speculation about whether they are looking.
7. Never state anything that is marked unverified. If a fact is only in the undated skills list, do not present it as an achievement.
8. Plain text, no emoji, no attachments, no links except the role link if provided.

## Skeleton

Subject: {{role}} at {{company}} - quick chat?

Hi {{CANDIDATE_FIRST_NAME}},

I noticed your work as {{title}} at {{employer}} ({{verified fact}}). We are hiring a {{role}} {{in location}} and that experience looks very relevant because {{one sentence}}.
{{optional: {{COLLEAGUE_NAME}}, who worked with you at {{shared employer}} in {{years}}, now works with us.}}
Would you be open to a 20-minute chat this week?

Thanks,
{{sender}}
