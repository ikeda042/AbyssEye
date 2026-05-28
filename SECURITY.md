# Security Policy

## Supported Versions

Until the first official release is tagged, security fixes should target the default branch. After releases begin, this file should be updated with the supported release lines.

## Reporting a Vulnerability

Do not open a public issue for suspected vulnerabilities.

Before publishing this repository as an official public project, replace this section with the approved security contact for the maintaining organization. A complete public release should provide:

- A monitored security email address or intake form
- Expected acknowledgement time
- Expected update cadence
- Disclosure policy
- PGP key or other secure reporting channel, if required by the organization

Temporary project-owner note: keep vulnerability reports private until the official reporting route is available.

## Deployment Assumptions

AbyssEye currently includes upload, model-management, file-download, and retraining workflows. The default application should be treated as a local or trusted-network tool unless an operator adds production authentication, authorization, logging, rate limits, storage quotas, and incident response procedures.

Public internet exposure without those controls is not recommended.

## Sensitive Data

Reports should not include microscopy data, model files, exported project ZIPs, private hostnames, internal IP addresses, credentials, or personally identifiable information. Use minimal reproduction steps and synthetic filenames whenever possible.
