# Project Page Framework

## Philosophy

Project pages should act as **invitations to explore**, not replicate documentation. They tell the story of why a project matters, what approach was taken, and what was learned—then direct readers to GitHub for technical details.

## Target Metrics

- **Length**: 100-150 lines of markdown (vs. 400-700+ lines of documentation)
- **Content ratio**: 80% narrative prose, 20% lists/bullets
- **Code blocks**: Zero (link to GitHub instead)
- **Sections**: 5-7 focused sections (vs. 15-30 exhaustive sections)

## The Transformation Pattern

### From Documentation To Narrative

**Documentation mindset** (what to avoid):
- "Here's how I built it" with code examples
- Comprehensive API references
- Exhaustive feature lists
- Mathematical proofs and formulas
- Architecture diagrams with implementation details

**Narrative mindset** (what to aim for):
- "Here's the problem this solves"
- Key insights and lessons learned
- Results and outcomes
- Critical design decisions explained in prose
- Why certain approaches were chosen

## Required Structure

### 1. Opening Hook (2-3 paragraphs)

**Purpose**: Immediately answer "why should I care?"

**Pattern**:
- Paragraph 1: State the problem or question that motivated the project
- Paragraph 2: Describe what the project does to address it
- Paragraph 3: Hint at the interesting finding or outcome

**Example** (from Kallos Portfolios):
> This is the question that quantitative researchers debate endlessly: can neural network forecasts translate into better investment decisions? Academic papers showcase impressive prediction accuracy, but in portfolio management, what matters is risk-adjusted returns after transaction costs. Kallos Portfolios was built to answer this question rigorously.

**Anti-pattern**:
> Kallos Portfolios is a sophisticated cryptocurrency portfolio optimization framework that systematically compares three distinct investment strategies...

### 2. The Approach (1-2 sections, 2-4 paragraphs each)

**Purpose**: Explain how you tackled the problem

**What to include**:
- High-level methodology
- Key design decisions
- What makes this approach different/better
- Critical constraints or requirements

**What to exclude**:
- Implementation details
- Code snippets
- Parameter tables
- Step-by-step procedures

**Example** (from Kallos Models):
> The system enforces separation between tuning, training, and evaluation to prevent data leakage. Step 1: Hyperparameter Tuning runs Optuna trials with walk-forward cross-validation. Step 2: Final Model Training uses optimal hyperparameters on combined data. Step 3: Hold-Out Evaluation assesses on completely unseen data.

### 3. Key Innovation or Insight (1-2 sections, 2-3 paragraphs each)

**Purpose**: Highlight what's interesting or novel

**Focus areas**:
- Technical innovations
- Methodological rigor
- Unexpected findings
- Production-ready practices
- Lessons learned

**Example** (from Kallos Data):
> Critical to reliability: every indicator calculation includes a warmup period. When calculating indicators for the past 60 days, the system loads 120 days of raw data, computes indicators across the full period, then trims the first 60 days to eliminate unstable initial values. This prevents artifacts from insufficient lookback windows that would bias model training.

### 4. Results or Outcomes (1 section, 2-3 paragraphs)

**Purpose**: What did you learn or achieve?

**Content**:
- Key findings (especially surprising ones)
- Performance results (concise, not exhaustive)
- Limitations and honest assessment
- What worked and what didn't

**Example** (from Kallos Portfolios):
> The t-test revealed the critical finding: while GRU portfolios achieved 4% higher Sharpe ratios than historical optimization (1.12 vs 1.08), the hypothesis test returned p = 0.18. There's an 18% probability this difference arose from random chance—far above the 5% threshold required to claim genuine outperformance.

### 5. Technologies (1 concise section)

**Format**:
```markdown
## Technologies

**Category 1**: Tool A, Tool B, Tool C

**Category 2**: Tool D, Tool E

**Category 3**: Tool F, Tool G
```

**Keep it to**: 3-4 categories, 2-4 tools per category

**Anti-pattern**: Long paragraphs explaining why each technology was chosen

### 6. Closing CTA (1 paragraph + link)

**Purpose**: Invite exploration of the repository

**Pattern**:
- 1-2 sentences summarizing what's in the repo
- Clear "View Repository →" link
- Optional: License info

**Example**:
> This framework demonstrates end-to-end quantitative portfolio management—from machine learning predictions through optimization, rigorous backtesting, and statistical validation. The complete implementation, including the inheritance-based simulator architecture, statistical testing suite, and async database operations, is available on GitHub.
>
> **[View Repository →](https://github.com/username/repo)**

## Writing Style Guidelines

### Tone

**Do**:
- Professional but accessible
- Evidence-based (what you actually built)
- Honest about limitations
- Engaging and story-driven

**Don't**:
- Overly academic or dry
- Aspirational (what you could build)
- Marketing-speak or hype
- Tutorial-style instructions

### Sentence Structure

**Prefer**:
- Active voice: "The system implements X" not "X is implemented"
- Concrete specifics: "20% lower prediction error" not "significantly better"
- Short paragraphs: 2-4 sentences maximum
- Scannable lists when appropriate

**Avoid**:
- Passive constructions
- Vague qualifiers ("very sophisticated," "highly optimized")
- Long, complex sentences
- Walls of text

### Technical Depth

**Right level**:
- Explain *what* and *why*, not *how*
- Include enough detail to demonstrate competence
- Use technical terms appropriately
- Assume reader has domain knowledge

**Example** (right level):
> The GRU prediction pipeline demonstrates critical operational realism often missing from academic research. Models train on quarterly windows and automatically deploy based on rebalancing dates. January rebalancing uses the Q4 2022 model. April uses Q1 2023.

**Example** (too detailed):
```python
class GRUPredictor:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        ...
```

**Example** (too shallow):
> The system uses machine learning to make predictions.

## Common Patterns by Project Type

### Machine Learning Projects

**Focus on**:
- Problem formulation and motivation
- Validation methodology (especially important)
- Interesting findings or performance
- Production practices (if applicable)

**De-emphasize**:
- Architecture diagrams
- Hyperparameter tables
- Training curves
- Code implementations

### Data Engineering Projects

**Focus on**:
- Reliability and error handling
- Scale (volume, velocity, variety)
- Integration with downstream systems
- Lessons from production operation

**De-emphasize**:
- Database schemas
- API client code
- ETL step-by-step procedures
- Configuration details

### Full-Stack Applications

**Focus on**:
- User problem being solved
- Key features and workflows
- Technical challenges overcome
- Design decisions

**De-emphasize**:
- Component architecture
- State management patterns
- API endpoints
- Deployment procedures

## Quality Checklist

Before publishing a project page, verify:

- [ ] **Opening hook** answers "why should I care?" in first paragraph
- [ ] **Length** is 100-150 lines (not 400+)
- [ ] **Code blocks** are zero (all code stays on GitHub)
- [ ] **Math formulas** are minimal (1-3 max, only if critical)
- [ ] **Narrative flow** tells a story, not a feature list
- [ ] **Evidence-based** content (what you actually built, not aspirational)
- [ ] **Results** include honest assessment and limitations
- [ ] **Technologies** section is concise (3-4 lines max)
- [ ] **GitHub CTA** is clear and compelling
- [ ] **Tone** is professional but engaging
- [ ] **Scannable** structure with clear section headers

## Template

```markdown
---
title: "Project Name: One-Line Description"
date: YYYY-MM-DD
category: Category
tags: [Tag1, Tag2, Tag3, Tag4, Tag5]
featured: true/false
order: N
image: /assets/project-image.png
excerpt: "Two-sentence summary for project cards. First sentence states what it does. Second sentence highlights key feature or finding."
github_url: https://github.com/username/repo
mathjax: true/false
---

## Opening Hook Title (Question or Statement)

[2-3 paragraphs establishing the problem, approach, and teaser of findings]

## The Approach or Methodology

[2-4 paragraphs explaining how you tackled the problem, key design decisions]

## Key Innovation or Technical Deep Dive

[2-3 paragraphs on what's interesting - technical innovation, methodological rigor, production practices]

## Results and Findings

[2-3 paragraphs on outcomes, what you learned, honest assessment]

## Optional: Additional Context Section

[If needed, 1-2 more sections on specific aspects worth highlighting]

## Technologies

**Category 1**: Tool A, Tool B, Tool C

**Category 2**: Tool D, Tool E

## Explore the Project

[1-2 sentences about what's in the repository]

**[View Repository →](https://github.com/username/repo)**

---

*Optional tagline summarizing the project's contribution*
```

## Examples

### ✅ Good Example (Kallos Portfolios Opening)

> Does Machine Learning Actually Improve Portfolio Construction?
>
> This is the question that quantitative researchers debate endlessly: can neural network forecasts translate into better investment decisions? Academic papers showcase impressive prediction accuracy, but in portfolio management, what matters is risk-adjusted returns after transaction costs. Kallos Portfolios was built to answer this question rigorously.
>
> The framework constructs and compares three cryptocurrency portfolios over 18 months: one using GRU neural network forecasts, one using traditional historical averages, and one passively tracking market-cap weights. All three employ identical optimization constraints, isolating the value of machine learning predictions from other portfolio construction decisions. Statistical hypothesis testing determines whether observed performance differences could have occurred by chance.

**Why it works**:
- Opens with compelling question
- Explains the motivation (prediction accuracy ≠ investment returns)
- Describes the approach clearly
- Hints at rigorous methodology

### ❌ Bad Example (Documentation Style)

> ## Overview
>
> Kallos Portfolios is a sophisticated cryptocurrency portfolio optimization framework that systematically compares three distinct investment strategies: neural network-based predictions, traditional mean-variance optimization, and passive market-cap weighting. The system integrates quarterly-trained GRU models with modern portfolio theory to answer a critical question: Does machine learning add value to quantitative portfolio construction?
>
> This framework represents the complete portfolio management lifecycle—from predictive modeling to optimization, backtesting, and rigorous statistical evaluation—designed for production deployment in cryptocurrency markets.

**Why it fails**:
- Starts with "Overview" (documentation red flag)
- Uses buzzwords ("sophisticated," "systematically")
- Lists features instead of telling a story
- Doesn't engage the reader emotionally

## Version History

- **v1.0** (2025-10-23): Initial framework based on Kallos project transformations
- Applied to: Kallos Portfolios, Kallos Models MLOps, Kallos Data Pipeline

## Usage

When creating or updating a project page:

1. Read this framework thoroughly
2. Use the template as starting structure
3. Follow the section-by-section guidance
4. Review against the quality checklist
5. Compare to good examples from existing pages
6. Aim for 100-150 lines total

**Remember**: Your goal is to make readers think "I want to explore this" not "I now understand every implementation detail."
