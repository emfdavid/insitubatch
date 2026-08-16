# Governance

## Statement of intent

insitubatch is today maintained by one person. **That is a transitional state, not the
goal.** The project is being run from v0.1.0 onward as a collaborative, multi-maintainer
effort, and this document is adopted *before* it is strictly needed — so that the second,
third and fourth maintainers arrive to a framework they can build on, rather than to an
absence that has to be filled in at the moment it is most likely to be contentious.

Two commitments follow from that, and they are meant to be held to:

- **The path in is merit-based and public.** Anyone who contributes sustained, quality work
  is eligible for the core developer group. There is no application, no minimum contribution
  count, and no requirement to be employed by anyone in particular.
- **Decisions are made in the open.** Non-sensitive project discussion happens on the issue
  tracker, not in private. A decision that cannot be explained on an issue is a decision that
  needs rethinking.

insitubatch also intends to seek **Zarr Affiliated Project** status. Of the Zarr Project's
three published criteria it already meets two — it is open source, and it is directly related
to Zarr (it is a consumer of, and contributor to, zarr-python and the wider ecosystem) — and
it is working on the third, a critical mass of sustained development activity. This is a
statement of direction, not a claim of existing affiliation; the governance below is
deliberately the Zarr Project's own template for affiliated projects, so that adopting it
later is a formality rather than a rewrite.

## Roles and responsibilities

### Users

Users are members of the community who use the project. Their contributions — feedback, bug
reports, and telling other people the project exists — are essential to its purpose and
direction. A good bug report is a real contribution and is treated as one.

### Contributors

Contributors are community members who engage directly with the project in concrete ways,
such as:

- Proposing, discussing, or reviewing a change to the code or documentation via a pull
  request.
- Reporting a GitHub issue.
- Assisting with documentation, examples, or project infrastructure.
- Supporting new users.

All community members are encouraged to contribute. See
[the contributing guide](https://emfdavid.github.io/insitubatch/contributing/) for how.
Contributions are made in compliance with the [Code of Conduct](CODE_OF_CONDUCT.md).

### Core developers

The Core Developers Group (CDG) is the governing and administrative body for the project.

- **Function.** Core developers have administrative rights and make decisions: accepting or
  rejecting pull requests, cutting releases, and managing the project's repositories
  (including adding and removing members). **A group of one is acceptable for a project this
  small** — it is where insitubatch is today.
- **Authority.** The CDG is self-governing.
- **Membership is merit-based.** Any contributor is eligible.
    - *Nomination.* Existing core developers can nominate new members. Nominations are based
      on clear evidence of **sustained, quality contribution** to the project — which
      explicitly includes review, documentation, triage and user support, not only merged
      code. Approval is by vote of the existing core developers: consensus ideally, majority
      approval at minimum.
    - *Removal.* Core developers who become inactive can and should be removed by majority
      vote. Core developers may resign at will. Neither is a judgement on the person; an
      accurate list of who is actually maintaining the project is a service to everyone
      relying on it.
- **Chair.** Larger projects should have a chair to coordinate and facilitate — a role with
  no additional authority. At the current size a chair is unnecessary, and none is appointed.

### Current core developers

- David Stuebe ([@emfdavid](https://github.com/emfdavid))

## Decision-making process

Decisions are made by consensus, following the
[Apache Software Foundation's model](https://community.apache.org/committers/decisionMaking.html).

**Lazy consensus** is the default. *"Essentially lazy consensus means that you don't need to
get explicit approval to proceed, but you need to be prepared to listen if someone objects."*
State your intent on a public GitHub issue; if no one objects within 72 hours, proceed. This
is appropriate for minor, non-controversial changes.

**Consensus building** is used for larger or more impactful decisions: discussion happens on
GitHub where community members can share feedback, sufficient time is given for objections to
be raised and defended, and the person who started the discussion posts a summary once
consensus appears to have been reached, so that their reading of it can be checked.

**A vote** is the last resort, called by a core developer on an issue or PR when consensus is
genuinely unreachable.

One project-specific exception. Changes to the **load-bearing invariants** — the ones listed
in [the contributing guide](https://emfdavid.github.io/insitubatch/contributing/) and the
scope limits in [DESIGN.md](DESIGN.md) — are **never** made by lazy consensus. They are the
decisions the whole design rests on, they are the ones most likely to be eroded by
accumulation rather than by argument, and they require explicit consensus among the core
developers with the reasoning written down.

## Code of Conduct

All participation in this project is governed by the [Code of Conduct](CODE_OF_CONDUCT.md),
which applies to maintainers first.

## License and attribution

This governance document is adapted from the Zarr Project's
[governance for affiliated software projects](https://github.com/zarr-developers/governance/blob/main/AFFILIATED_PROJECT_GOVERNANCE.md),
which is itself adapted from the original Zarr governance document and the
[Meritocratic governance model](http://oss-watch.ac.uk/resources/meritocraticgovernancemodel)
by Ross Gardler and Gabriel Hanganu, licensed under a Creative Commons Attribution-ShareAlike
4.0 International License.
