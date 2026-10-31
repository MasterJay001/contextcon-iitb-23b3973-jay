"""Synthetic labour market with PLANTED hiring effects - the offline stand-in for Crustdata.

Why it exists: (1) tests and CI run with no network/keys, (2) it lets us verify the *evaluation
harness itself*: in an `effect` world (hires depend on fit + movability + connection) the harness
must detect A4 > A1, and in a `null` world (hires depend on fit only) it must report no clear
gain. A harness that cannot recover a planted effect cannot be trusted on real data.

Results produced from this world are SYNTHETIC. They verify code, never claims (CLAUDE.md #14).

Generative story
  * people follow careers with log-logistic job spells (hazard rises then falls: the "movement
    peaks after ~18-36 months" hypothesis is true here by construction, with person-level
    heterogeneity), promotions inside a company, and job changes between companies;
  * postings appear at companies at date T; for each, hires are drawn (softmax) among people who
    are employed elsewhere at T with
        logit = bF*fit + bM*P(move within 6 months | tenure, person scale) + bC*connection(ex-colleagues at the company)
  * the careers are then rewritten so the hire leaves their old job and joins the company a few
    weeks/months after T, and everyone's career continues to "now" (so today's profile of a hire shows
    the new job: the leak the Time Guard must close).
"""
from __future__ import annotations

import datetime as dt
import math
import random
from dataclasses import dataclass, field

from core.models import seniority_label
from crust.titles import family_distance, normalize_title

NOW = dt.date(2026, 10, 31)
DAYS_PER_MONTH = 30.4375

CITIES = [("Bengaluru", "Karnataka", 0.34), ("Mumbai", "Maharashtra", 0.20), ("Pune", "Maharashtra", 0.12),
          ("Hyderabad", "Telangana", 0.12), ("Gurugram", "Haryana", 0.10), ("Chennai", "Tamil Nadu", 0.07),
          ("Noida", "Uttar Pradesh", 0.05)]
INDUSTRIES = {
    "Financial Services": ["payments", "ledger", "UPI", "lending", "fraud detection", "card issuing"],
    "Software Development": ["SaaS", "multi-tenant APIs", "developer tools", "workflow automation"],
    "E-commerce": ["catalog", "checkout", "order management", "marketplace"],
    "Hospitals and Health Care": ["patient records", "telemedicine", "claims"],
    "Transportation, Logistics": ["routing", "fleet", "last-mile delivery"],
    "Education": ["learning platform", "assessments"],
    "Media and Telecommunications": ["streaming", "ad tech"],
}
SCHOOLS = ["IIT Bombay", "IIT Delhi", "BITS Pilani", "NIT Trichy", "IIIT Hyderabad", "VIT Vellore", "Anna University",
           "Delhi University", "Pune University", "Manipal Institute of Technology"]

FAMILIES = {
    # weight, nouns, skills, department
    "swe_backend": (0.20, ["Backend Engineer", "Backend Developer", "Software Engineer - Backend", "Software Engineer"],
                    ["Go", "Java", "Python", "Kafka", "PostgreSQL", "Microservices", "Redis", "gRPC", "Distributed Systems", "AWS"]),
    "swe_general": (0.12, ["Software Engineer", "Software Developer", "Member of Technical Staff"],
                    ["Java", "Python", "JavaScript", "SQL", "REST APIs", "Git", "Docker"]),
    "swe_frontend": (0.09, ["Frontend Engineer", "Frontend Developer", "UI Engineer"], ["React", "TypeScript", "CSS", "Next.js", "Webpack"]),
    "swe_fullstack": (0.08, ["Full Stack Engineer", "Full Stack Developer"], ["React", "Node.js", "TypeScript", "PostgreSQL", "AWS"]),
    "swe_mobile": (0.04, ["Android Developer", "iOS Engineer", "Mobile Engineer"], ["Kotlin", "Swift", "Flutter", "React Native"]),
    "data_engineer": (0.07, ["Data Engineer", "Analytics Engineer"], ["Spark", "Airflow", "SQL", "Python", "Snowflake", "Kafka"]),
    "devops_sre": (0.05, ["DevOps Engineer", "Site Reliability Engineer"], ["Kubernetes", "Terraform", "AWS", "Prometheus", "CI/CD"]),
    "ml_engineer": (0.04, ["Machine Learning Engineer", "AI Engineer"], ["PyTorch", "TensorFlow", "NLP", "MLOps", "Python"]),
    "data_scientist": (0.04, ["Data Scientist"], ["Python", "Statistics", "SQL", "Experimentation", "scikit-learn"]),
    "qa_engineer": (0.04, ["QA Engineer", "SDET"], ["Selenium", "Test Automation", "Java", "Cypress"]),
    "product_manager": (0.07, ["Product Manager"], ["Roadmapping", "SQL", "A/B Testing", "User Research", "PRDs"]),
    "product_designer": (0.04, ["Product Designer", "UX Designer"], ["Figma", "Prototyping", "User Research", "Design Systems"]),
    "growth_marketing": (0.05, ["Growth Marketing Manager", "Performance Marketer"], ["SEO", "Google Ads", "Funnel Analytics", "CRM"]),
    "account_executive": (0.04, ["Account Executive", "Sales Executive"], ["Enterprise Sales", "Negotiation", "Salesforce", "Prospecting"]),
    "customer_success": (0.03, ["Customer Success Manager"], ["Onboarding", "Renewals", "Zendesk", "Account Management"]),
}
LEVEL_PREFIX = {0: ["Intern"], 1: ["Junior", "Associate"], 2: [""], 3: ["Senior"], 4: ["Staff", "Lead"], 5: ["Principal"]}
FIRST = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ishaan", "Krishna", "Rohan", "Ananya", "Diya", "Isha",
         "Kavya", "Meera", "Priya", "Riya", "Saanvi", "Tara", "Neha", "Karan", "Nikhil", "Pooja", "Rahul", "Sneha", "Varun", "Zoya"]
LAST = ["Sharma", "Verma", "Iyer", "Nair", "Reddy", "Gupta", "Patel", "Mehta", "Kulkarni", "Joshi", "Menon", "Rao", "Das", "Khan",
        "Singh", "Bose", "Chopra", "Desai", "Pillai", "Kapoor"]
SYL = ["zen", "nim", "quil", "vor", "lum", "tra", "kes", "dyn", "orb", "fyn", "pay", "stack", "grid", "loop", "wave", "forge",
       "ark", "mint", "pulse", "nova", "crest", "tide"]
SUFFIX = ["Labs", "Pay", "Systems", "Works", "Cloud", "AI", "Tech", "Logic", "Health", "Mart"]


@dataclass
class SimRole:
    company_id: int
    title: str
    family: str
    level: int
    start: dt.date
    end: dt.date | None
    city: str
    desc: str = ""


@dataclass
class SimPerson:
    pid: int
    first: str
    last: str
    home_city: str
    alpha: float          # person-level median tenure scale (months)
    family: str
    skills: list[str]
    school: str
    career_start: dt.date
    roles: list[SimRole] = field(default_factory=list)


@dataclass
class SimCompany:
    cid: int
    name: str
    slug: str
    industry: str
    city: str
    state: str
    size: int
    target_like: bool

    @property
    def url(self) -> str:
        return f"https://www.linkedin.com/company/{self.slug}"

    @property
    def size_range(self) -> str:
        for hi, label in [(10, "2-10"), (50, "11-50"), (200, "51-200"), (500, "201-500"), (1000, "501-1000"), (5000, "1001-5000"),
                          (10000, "5001-10000")]:
            if self.size <= hi:
                return label
        return "10001+"


@dataclass
class SimPosting:
    job_id: int
    company_id: int
    T: dt.date
    family: str
    level: int
    city: str
    title: str
    description: str
    hires: list[int] = field(default_factory=list)


def _months_to_days(m: float) -> int:
    return max(1, int(round(m * DAYS_PER_MONTH)))


def log_logistic_survival(t: float, alpha: float, beta: float) -> float:
    return 1.0 / (1.0 + (max(t, 0.0) / alpha) ** beta)


class SimWorld:
    def __init__(self, seed: int = 7, n_people: int = 4000, n_target: int = 40, n_other: int = 60, *,
                 effect: str = "effect", betas: tuple[float, float, float] | None = None, now: dt.date = NOW,
                 deny_skills: bool = False, shape: float = 2.2):
        self.seed, self.now, self.shape = seed, now, shape
        self.deny_skills = deny_skills
        if betas is None:
            betas = (4.0, 5.0, 3.0) if effect == "effect" else (4.0, 0.0, 0.0)
        self.betas = betas
        self.rng = random.Random(seed)
        self.companies: dict[int, SimCompany] = {}
        self.people: dict[int, SimPerson] = {}
        self.postings: list[SimPosting] = []
        self._gen_companies(n_target, n_other)
        self._gen_people(n_people)
        self._gen_postings()
        self._payloads: dict[int, dict] = {}
        for pid in self.people:
            self._payloads[pid] = self._build_payload(pid)

    # ------------------------------------------------------------------------ generation
    def _pick_city(self) -> tuple[str, str]:
        r, acc = self.rng.random(), 0.0
        for city, state, w in CITIES:
            acc += w
            if r <= acc:
                return city, state
        return CITIES[0][0], CITIES[0][1]

    def _gen_companies(self, n_target: int, n_other: int) -> None:
        used: set[str] = set()
        for i in range(n_target + n_other):
            while True:
                nm = (self.rng.choice(SYL) + self.rng.choice(SYL)).capitalize() + " " + self.rng.choice(SUFFIX)
                if nm not in used:
                    used.add(nm)
                    break
            target = i < n_target
            size = int(math.exp(self.rng.uniform(math.log(60), math.log(4500)))) if target else int(math.exp(self.rng.uniform(math.log(15), math.log(150000))))
            city, state = self._pick_city()
            ind = self.rng.choice(list(INDUSTRIES))
            cid = 100000 + i
            self.companies[cid] = SimCompany(cid, nm, nm.lower().replace(" ", "-"), ind, city, state, size, target)
        self._weights = {cid: (1.0 / math.log(2 + c.size)) * (1.4 if c.target_like else 1.0) for cid, c in self.companies.items()}

    def _title(self, family: str, level: int) -> str:
        noun = self.rng.choice(FAMILIES[family][1])
        if level == 0:
            return f"{noun} Intern"
        pre = self.rng.choice(LEVEL_PREFIX[level])
        return f"{pre} {noun}".strip()

    def _desc(self, family: str, company: SimCompany, skills: list[str]) -> str:
        dom = self.rng.choice(INDUSTRIES[company.industry])
        sk = ", ".join(self.rng.sample(skills, k=min(3, len(skills))))
        verb = self.rng.choice(["Built", "Owned", "Designed", "Scaled", "Shipped", "Maintained"])
        return f"{verb} {dom} systems using {sk}. Worked with cross-functional teams at {company.name}."

    def _spell_months(self, alpha: float) -> float:
        u = min(max(self.rng.random(), 1e-4), 1 - 1e-4)
        return max(3.0, min(96.0, alpha * (u / (1 - u)) ** (1.0 / self.shape)))

    def _choose_company(self, avoid: int | None, industry_hint: str | None) -> int:
        ids = [c for c in self.companies if c != avoid]
        w = []
        for c in ids:
            base = self._weights[c]
            if industry_hint and self.companies[c].industry == industry_hint:
                base *= 2.5
            w.append(base)
        return self.rng.choices(ids, weights=w, k=1)[0]

    def _extend_career(self, p: SimPerson, role: SimRole) -> None:
        """Continue a career from `role` (already appended, end unset) until `now`."""
        cur = role
        while True:
            dur = self._spell_months(p.alpha)
            end = cur.start + dt.timedelta(days=_months_to_days(dur))
            if end >= self.now:
                cur.end = None
                return
            cur.end = end
            if self.rng.random() < 0.30 and cur.level < 5:      # promotion inside the company
                lvl = cur.level + 1
                nxt = SimRole(cur.company_id, self._title(p.family, lvl), p.family, lvl, end, None, cur.city,
                              self._desc(p.family, self.companies[cur.company_id], p.skills))
            else:
                if self.rng.random() < 0.08:                    # occasional lateral family switch
                    p.family = self.rng.choice(list(FAMILIES))
                gap = self.rng.choice([0, 0, 1, 1, 2, 3, 6])
                start = end + dt.timedelta(days=_months_to_days(gap))
                if start >= self.now:
                    return
                lvl = min(5, max(1, cur.level + self.rng.choices([-1, 0, 1], weights=[0.08, 0.5, 0.42])[0]))
                comp = self._choose_company(cur.company_id, self.companies[cur.company_id].industry if self.rng.random() < 0.6 else None)
                c = self.companies[comp]
                city = c.city if self.rng.random() < 0.8 else p.home_city
                nxt = SimRole(comp, self._title(p.family, lvl), p.family, lvl, start, None, city, self._desc(p.family, c, p.skills))
            p.roles.append(nxt)
            cur = nxt

    def _gen_people(self, n: int) -> None:
        fam_ids = list(FAMILIES)
        fam_w = [FAMILIES[f][0] for f in fam_ids]
        for pid in range(1_000_001, 1_000_001 + n):
            fam = self.rng.choices(fam_ids, weights=fam_w, k=1)[0]
            city, _ = self._pick_city()
            start_year = self.rng.randint(2008, 2024)
            cstart = dt.date(start_year, self.rng.randint(1, 12), self.rng.randint(1, 28))
            if cstart >= self.now - dt.timedelta(days=120):
                cstart = self.now - dt.timedelta(days=400)
            skills = self.rng.sample(FAMILIES[fam][2], k=min(len(FAMILIES[fam][2]), self.rng.randint(3, 6)))
            p = SimPerson(pid, self.rng.choice(FIRST), self.rng.choice(LAST), city, alpha=math.exp(self.rng.gauss(math.log(30), 0.35)),
                          family=fam, skills=skills, school=self.rng.choice(SCHOOLS), career_start=cstart)
            comp = self._choose_company(None, None)
            c = self.companies[comp]
            first = SimRole(comp, self._title(fam, 1 if self.rng.random() < 0.7 else 2), fam, 1, cstart, None, c.city if self.rng.random() < 0.8 else city,
                            self._desc(fam, c, skills))
            first.level = 1 if "Junior" in first.title or "Associate" in first.title else 2
            p.roles.append(first)
            self._extend_career(p, first)
            self.people[pid] = p

    # ------------------------------------------------------------------------ queries on truth
    def role_at(self, p: SimPerson, d: dt.date) -> SimRole | None:
        for r in p.roles:
            if r.start <= d and (r.end is None or r.end > d):
                return r
        return None

    def employed_at(self, company_id: int, d: dt.date) -> list[int]:
        out = []
        for p in self.people.values():
            r = self.role_at(p, d)
            if r and r.company_id == company_id:
                out.append(p.pid)
        return out

    def _conn_raw(self, p: SimPerson, roster_roles: dict[int, list[SimRole]], T: dt.date) -> float:
        """Sum over roster members of overlap-months at shared past employers, inverse-log size weighted."""
        x = 0.0
        for r in p.roles:
            if r.start >= T:
                continue
            r_end = r.end or T
            r_end = min(r_end, T)
            for rr in roster_roles.get(r.company_id, []):
                lo, hi = max(r.start, rr.start), min(r_end, min(rr.end or T, T))
                months = (hi - lo).days / DAYS_PER_MONTH
                if months >= 3:
                    x += months / math.log(2 + self.companies[r.company_id].size)
        return x

    def _fit(self, role: SimRole, target_family: str, target_level: int, target_city: str) -> float:
        fd = family_distance(role.family, target_family)
        fam = {0: 1.0, 1: 0.6}.get(fd, 0.1)
        dl = role.level - target_level
        sen = {0: 1.0, -1: 0.8, 1: 0.5}.get(dl, 0.1)
        loc = 1.0 if role.city == target_city else 0.45
        return 0.5 * fam + 0.3 * sen + 0.2 * loc

    def _gen_postings(self) -> None:
        targets = [c for c in self.companies.values() if c.target_like]
        window_days = (dt.date(2025, 9, 30) - dt.date(2025, 4, 1)).days
        jid = 7_000_001
        fam_ids = [f for f in FAMILIES if f.startswith(("swe", "data", "devops", "ml", "qa"))]
        for c in targets:
            T = dt.date(2025, 4, 1) + dt.timedelta(days=self.rng.randint(0, window_days))
            fam = self.rng.choice(fam_ids)
            level = self.rng.choices([2, 3, 4], weights=[0.3, 0.5, 0.2])[0]
            title = self._title(fam, level)
            skills = FAMILIES[fam][2]
            desc = (f"We are hiring a {title} in {c.city}. You will build {self.rng.choice(INDUSTRIES[c.industry])} systems. "
                    f"Requirements: {', '.join(self.rng.sample(skills, k=min(4, len(skills))))}. {self.rng.randint(3, 8)}+ years of experience.")
            post = SimPosting(jid, c.cid, T, fam, level, c.city, title, desc)
            jid += 1
            self.postings.append(post)
        self.postings.sort(key=lambda x: (x.T, x.job_id))
        for post in self.postings:
            self._apply_hiring(post)

    def _apply_hiring(self, post: SimPosting) -> None:
        T = post.T
        roster = self.employed_at(post.company_id, T)
        roster_roles: dict[int, list[SimRole]] = {}
        for pid in roster:
            for r in self.people[pid].roles:
                if r.start < T:
                    roster_roles.setdefault(r.company_id, []).append(r)
        cands = []
        for p in self.people.values():
            r = self.role_at(p, T)
            if not r or r.company_id == post.company_id:
                continue
            if family_distance(r.family, post.family) > 1 or abs(r.level - post.level) > 2:
                continue
            t_months = (T - r.start).days / DAYS_PER_MONTH
            s0 = log_logistic_survival(t_months, p.alpha, self.shape)
            s1 = log_logistic_survival(t_months + 6, p.alpha, self.shape)
            mov = 1.0 - s1 / s0
            raw = self._conn_raw(p, roster_roles, T)
            conn = raw / (raw + 3.0)
            fit = self._fit(r, post.family, post.level, post.city)
            cands.append((p.pid, self.betas[0] * fit + self.betas[1] * mov + self.betas[2] * conn))
        if not cands:
            return
        n_hires = self.rng.choices([1, 2, 3], weights=[0.7, 0.2, 0.1])[0]
        chosen: list[int] = []
        pool = list(cands)
        for _ in range(min(n_hires, len(pool))):
            m = max(l for _, l in pool)
            w = [math.exp(l - m) for _, l in pool]
            idx = self.rng.choices(range(len(pool)), weights=w, k=1)[0]
            chosen.append(pool.pop(idx)[0])
        comp = self.companies[post.company_id]
        for pid in chosen:
            p = self.people[pid]
            start_new = T + dt.timedelta(days=self.rng.randint(21, 140))
            old = self.role_at(p, T)
            keep = [r for r in p.roles if r.start <= T]
            if old is not None:
                old.end = start_new - dt.timedelta(days=self.rng.randint(0, 20))
                if old.end <= old.start:
                    old.end = old.start + dt.timedelta(days=30)
            p.roles = keep
            role = SimRole(post.company_id, post.title, post.family, post.level, start_new, None, comp.city,
                           self._desc(post.family, comp, p.skills))
            p.roles.append(role)
            self._extend_career(p, role)
            p.family = post.family
        post.hires = chosen

    # ------------------------------------------------------------------------ payloads (today's snapshot)
    @staticmethod
    def _degrade(d: dt.date | None, rng: random.Random) -> str | None:
        """Real data is fuzzy: month- and year-level dates arrive as YYYY-MM-01 / YYYY-01-01."""
        if d is None:
            return None
        r = rng.random()
        if r < 0.55:
            return dt.date(d.year, d.month, 1).isoformat()
        if r < 0.85:
            return d.isoformat()
        return dt.date(d.year, 1, 1).isoformat()

    def _role_payload(self, p: SimPerson, r: SimRole, i: int) -> dict:
        c = self.companies[r.company_id]
        rng = random.Random(f"{self.seed}-{p.pid}-{i}")
        nt = normalize_title(r.title)
        return {
            "name": c.name, "professional_network_id": str(c.cid), "title": r.title, "description": r.desc,
            "location": {"raw": f"{r.city}, India"}, "employment_type": "full_time",
            "start_date": self._degrade(r.start, rng), "end_date": self._degrade(r.end, rng),
            "is_default": r.end is None, "crustdata_company_id": c.cid,
            "company_professional_network_profile_url": c.url, "seniority_level": seniority_label(r.level),
            "function_category": nt.department, "years_at_company_raw": round(((r.end or self.now) - r.start).days / 365.25),
            "company_headcount_latest": c.size, "company_headcount_range": c.size_range, "company_industries": [c.industry],
            "company_type": "Privately Held", "company_status": "active", "company_headquarters_country": "IND",
            "company_hq_location": f"{c.city}, {c.state}, India",
        }

    def _build_payload(self, pid: int) -> dict:
        p = self.people[pid]
        roles = sorted(p.roles, key=lambda r: r.start, reverse=True)
        current = [self._role_payload(p, r, i) for i, r in enumerate(roles) if r.end is None]
        past = [self._role_payload(p, r, 100 + i) for i, r in enumerate(roles) if r.end is not None]
        top = roles[0]
        c = self.companies[top.company_id]
        city_state = {n: s for n, s, _ in CITIES}
        nt = normalize_title(top.title)
        grad = p.career_start.year - self.rng.randint(0, 1)
        yrs = round((self.now - p.career_start).days / 365.25, 1)
        recent = bool(top.start and (self.now - top.start).days <= 120)
        return {
            "crustdata_person_id": pid,
            "basic_profile": {
                "name": f"{p.first} {p.last}", "first_name": p.first, "last_name": p.last,
                "headline": f"{top.title} at {c.name}", "current_title": top.title,
                "summary": f"{top.title} at {c.name}. Previously across {len(roles)} roles.",
                "location": {"city": top.city, "state": city_state.get(top.city, ""), "country": "India", "continent": "Asia",
                             "raw": f"{top.city}, India", "full_location": f"{top.city}, {city_state.get(top.city, '')}, India"},
                "normalized_title": {"matched_title": nt.matched_title, "department": nt.department, "sub_department": nt.sub_department,
                                     "similarity": nt.similarity, "confident": nt.confident},
            },
            "experience": {"employment_details": {"current": current, "past": past}, "recently_changed_jobs": recent},
            "education": {"schools": [{"school": p.school, "degree": "B.Tech", "field_of_study": "Computer Science",
                                       "start_year": grad - 4, "end_year": grad}]},
            "skills": {"professional_network_skills": list(p.skills)},
            "social_handles": {"professional_network_identifier": {"profile_url": f"https://www.linkedin.com/in/{p.first.lower()}-{p.last.lower()}-{pid}"}},
            "years_of_experience_raw": yrs,
            "metadata": {"updated_at": self.now.isoformat() + "T00:00:00"},
        }

    def person_payload(self, pid: int) -> dict:
        return self._payloads[pid]

    def job_payload(self, post: SimPosting) -> dict:
        c = self.companies[post.company_id]
        state = {n: s for n, s, _ in CITIES}.get(post.city, "")
        return {
            "crustdata_job_id": post.job_id,
            "job_details": {"job_id": post.job_id, "title": post.title, "category": "Engineering", "workplace_type": "Hybrid",
                            "source": "professional_network", "employment_type": "full_time", "url": f"https://jobs.example/{post.job_id}"},
            "company": {"basic_info": {"crustdata_company_id": c.cid, "name": c.name, "primary_domain": f"{c.slug}.com",
                                       "professional_network_id": str(c.cid), "industries": [c.industry]},
                        "headcount": {"total": c.size, "range": c.size_range}},
            "location": {"raw": f"{post.city}, {state}, India", "city": post.city, "state": state, "country": "India"},
            "content": {"description": post.description},
            "metadata": {"date_added": post.T.isoformat() + "T09:00:00", "date_updated": post.T.isoformat() + "T09:00:00"},
        }

    def truth(self) -> dict[int, list[int]]:
        return {p.job_id: list(p.hires) for p in self.postings}
