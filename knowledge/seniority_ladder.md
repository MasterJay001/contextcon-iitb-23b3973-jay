# Seniority ladder used by the normaliser (0-8)

| Level | Label | Typical title cues |
|---|---|---|
| 0 | Intern | intern, trainee, apprentice |
| 1 | Junior | junior, associate (as a prefix), graduate, entry level, SDE-1, Engineer I, APM |
| 2 | Mid | no modifier, II, 2, SDE-2, "Software Engineer", "Product Manager" |
| 3 | Senior | senior, sr, III, SDE-3, "founding engineer" (minimum) |
| 4 | Lead | lead, staff, tech lead, team lead, engineering manager (people manager), supervisor |
| 5 | Principal | principal, distinguished, fellow, senior manager, group manager, associate director |
| 6 | Director | director, head of X, AVP (assistant vice president) |
| 7 | VP | vice president, VP, SVP, EVP |
| 8 | C-level | CEO, CTO, CFO, COO, founder, co-founder, managing director, owner |

Rules of thumb used in scoring:

- Hires usually move sideways or one level up. A candidate one level BELOW the target is a promotion hire and is still plausible; one level ABOVE is
  usually overqualified and a weaker fit.
- "Manager" means a people-manager (Lead) for engineering, sales, operations, marketing and HR, but NOT for Product Manager, Program Manager, Account Manager
  or Customer Success Manager, which are individual-contributor tracks.
- "Associate" before a role means junior (Associate Product Manager); before director/vice president it means one step below.
- When a title carries no seniority cue, the API's role-level `seniority_level` label is used as a hint.
