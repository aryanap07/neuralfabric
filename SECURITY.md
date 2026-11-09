# Security Policy

NeuralFabric takes the security of its users, contributors, and distribution process seriously. This document explains which versions receive security attention, how to report a vulnerability, and what to expect during coordinated disclosure.

## Supported Versions

Security fixes are prioritised for:

- The latest published release of NeuralFabric.
- The current `main` branch.

Users are encouraged to upgrade to the latest release before reporting an issue. Older releases and unmaintained branches may not receive security updates. As NeuralFabric is under active development, supported versions may change as the project evolves.

## Reporting a Vulnerability

**Please do not disclose suspected security vulnerabilities through public GitHub issues, pull requests, discussions, or other public channels.** Public disclosure before a fix is available may put users at risk.

Please use one of the following private channels:

1. **GitHub private vulnerability reporting:** Open the repository's **Security** tab and use **Report a vulnerability**, when that option is available.
2. **Email:** Send the report to [aryanpatel2906.ap@gmail.com](mailto:aryanpatel2906.ap@gmail.com) with the subject line `Security Report: NeuralFabric`.

If GitHub private vulnerability reporting is unavailable, use the email address above. Please avoid sending confidential datasets, credentials, private keys, access tokens, or other sensitive material. Redact them from logs and proof-of-concept files before sharing.

### What to Include

A useful report should include as much of the following information as possible:

- A concise description of the suspected vulnerability and its potential impact.
- The affected NeuralFabric version, release, or commit.
- The Python version, operating system, and relevant dependency versions.
- The affected module, function, or API, if known.
- Clear, minimal steps to reproduce the issue.
- A proof of concept or test case that demonstrates the security impact, where safe to provide.
- Any known prerequisites, workarounds, or mitigations.

Please keep proof-of-concept examples minimal and avoid accessing, modifying, or disclosing data that does not belong to you.

## What to Expect

After receiving a report, the maintainers will make a reasonable effort to:

1. Acknowledge the report within **7 calendar days**.
2. Review the reported behavior and determine whether it affects NeuralFabric.
3. Ask for additional information when needed and keep the reporter informed of material status changes.
4. Develop and test an appropriate fix, where the issue is confirmed.
5. Coordinate publication of a security advisory and release information after a fix or mitigation is available.

These are response goals, not a guaranteed resolution deadline. The time required to investigate and release a fix depends on the complexity, severity, and maintainers' availability.

We appreciate responsible disclosure. Please allow maintainers a reasonable opportunity to investigate and address a confirmed issue before sharing technical details publicly. Reporters will be credited in an advisory or release note when they request recognition and it is appropriate to do so.

## Scope

Security reports are relevant when they demonstrate a practical impact on NeuralFabric or its supported development and distribution processes. Examples include:

- Arbitrary code execution caused by NeuralFabric's own implementation or its documented interfaces.
- Unintended access to, modification of, or disclosure of files or data through the package.
- Dependency or packaging issues that can compromise users of the published package.
- Compromise or unintended disclosure of credentials, release secrets, or other sensitive project data.
- A vulnerability in the GitHub Actions workflows that could compromise the repository or its release process.
- A denial-of-service condition caused by a realistic input within documented usage, where the impact is materially beyond normal resource consumption.

NeuralFabric is currently distributed as a local Python machine-learning library. It does not define a hosted inference service or a network-facing API. Reports should therefore focus on the package, its dependencies, repository automation, or distribution process rather than unrelated infrastructure.

### Generally Out of Scope

The following are generally not considered security vulnerabilities on their own:

- Incorrect predictions, model accuracy limitations, or expected numerical and statistical behavior.
- Adversarial examples, data poisoning, model bias, or model robustness findings that do not demonstrate a security impact on the software, its users' data, or its execution environment.
- Resource use resulting from extremely large or intentionally unreasonable input sizes, unless it reveals a disproportionate denial-of-service condition in documented usage.
- Vulnerabilities in unsupported versions or environments that cannot be reproduced on a supported configuration.
- Issues in third-party packages that do not create a demonstrable security impact through NeuralFabric. Relevant dependency vulnerabilities may still be reported for assessment.
- Findings that require compromising a user's device, account, or unrelated third-party service first.

An issue being out of scope does not mean it is unimportant; it may be better suited to a regular bug report or feature discussion. When uncertain, report it privately and let the maintainers assess it.

## Safe Research Guidelines

When investigating or validating a suspected vulnerability, please:

- Test only against your own local environment and data, or systems for which you have explicit permission.
- Avoid destructive testing, service disruption, denial-of-service attacks, social engineering, and attempts to access another person's data.
- Do not modify or delete repository data, publish malicious packages, or interfere with the project's release process.
- Collect only the minimum evidence needed to demonstrate the issue.
- Stop testing and report the issue privately if you encounter sensitive data or unexpected access to resources outside your control.

The maintainers support good-faith security research conducted within these guidelines and applicable law. This policy is not permission to test systems or services operated by third parties.

## Security Considerations for Users

NeuralFabric is an evolving research and educational framework. Users should apply appropriate safeguards when using it in experiments or applications:

- Treat external datasets and user-provided inputs as untrusted. Validate input shape, type, dimensions, and size before processing.
- Use trusted sources for Python packages and dependencies, and keep the environment updated.
- Avoid placing credentials, private datasets, or other sensitive information in public notebooks, examples, issue reports, or logs.
- Review code and dependencies before executing them, especially when working with data or environments that require confidentiality.
- Independently evaluate model behavior and application-level safeguards before using outputs in consequential settings.

NeuralFabric is provided under the terms of its [MIT License](LICENSE). The project does not make a security or suitability guarantee for a particular deployment. Users remain responsible for assessing and securing the systems in which they use the library.

## Security Updates

Confirmed vulnerabilities and relevant fixes will be communicated through GitHub Security Advisories, release notes, or both, as appropriate. Users should monitor the repository and upgrade to the latest supported release to receive available fixes.

Thank you for helping keep NeuralFabric and its community secure.
