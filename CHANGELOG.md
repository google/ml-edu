# Changelog

<!--

Changelog follow the https://keepachangelog.com/ standard (at least the headers)

This allow to:

* auto-parsing release notes during the automated releases from github-action:
  https://github.com/marketplace/actions/pypi-github-auto-release
* Have clickable headers in the rendered markdown

To release a new version (e.g. from `1.0.0` -> `2.0.0`):

* Create a new `# [2.0.0] - YYYY-MM-DD` header and add the current
  `[Unreleased]` notes.
* At the end of the file:
  * Define the new link url:
  `[2.0.0]: https://github.com/google/ml-edu/compare/v1.0.0...v2.0.0`
  * Update the `[Unreleased]` url: `v1.0.0...HEAD` -> `v2.0.0...HEAD`

-->

## [Unreleased]

## [0.1.3] - 2025-07-08

* Make classification threshold optional so experiment settings can also be used with regression.
* Add method to plot model predictions for one and two input features.

## [0.1.2] - 2025-02-19

* Fix experiment evaluation to return the correct metric values.

## [0.1.1] - 2025-02-06

* First upload to PyPi with basic classes and tests.

## [0.1.0] - 2024-07-12

* Initial release with project skeleton

<!-- mdlint off(LINK_UNUSED_ID) -->
[Unreleased]: https://github.com/google/ml-edu/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/google/ml-edu/releases/tag/v0.1.3
[0.1.2]: https://github.com/google/ml-edu/releases/tag/v0.1.2
[0.1.1]: https://github.com/google/ml-edu/releases/tag/v0.1.1
[0.1.0]: https://github.com/google/ml-edu/releases/tag/v0.1.0
