# Contribution Process and Specifications

<br>

Thank you for considering contributing! We welcome contributions in any form, including bug fixes, feature enhancements, documentation improvements, and even simple usage feedback.
Whether you're an experienced developer or a newcomer to an open source project for the first time, your contributions will be invaluable.

You can support this project in the following ways:

 * **Code Contribution: Fix known bugs, optimize performance, refactor code, or implement new features.** Follow the [1. Contribution Process](#1-contribution-process) and the [2. Code Specifications](#2-code-specifications) when submitting code.
 * **Issue Feedback: Report bugs through issue, raise function suggestions or questions, or participate in requirement review and solution discussion.**
 * **Document Improvement: Correct document errors, supplement missing content, optimize descriptions, or write examples and tutorials. Follow the [3. Document Specifications](#3-document-specifications).**
 * **Quality Assurance: Supplement or optimize test cases, review Pull Requests, provide constructive suggestions, and assist other contributors in improving code quality.**
 * **Community Promotion: Answer questions, share usage experience, and best practices in the issue/PR, or write blogs, tutorials, and promote projects by publicizing them in technical communities and social media.**

## 1. Contribution Process

1. **Derived Repository: Fork the source code repository to a personal repository, and then clone the personal repository to the local development environment.**
2. **Creating a Branch: Create a functional branch based on the latest main branch. The branch name should be concise and reflect the changes (for example, `fix_xxx_bug`, `feature_xxx`).**
3. **Code Development: Perform development on the function branch.** Follow the [2. Code Specifications](#2-code-specifications) and keep submission records clear and atomic.
4. **Local Test: Verify the functions of the code and supplement unit tests as required based on the development module situation to ensure that all tests are passed.**
5. **Document Update: Supplement or update the documents related to the change.** Follow the [3. Document Specifications](#3-document-specifications).
6. **Request to merge: Submit a PR and comply with the [4. Pull Request Specifications](#4-pull-request-specifications). For details, see [5. Pull Request Process Description](#5-pull-request-process-description).**
7. **Tracking and Incorporation: Track the Pull Request progress, respond to review comments in a timely manner, and modify the code until the code passes the review and is incorporated into the mainline.**

<br>

## 2. Code Specifications

### 2.1 Python Code Specifications

 * **Coding Specifications: Comply with [PEP 8](https://peps.python.org/pep-0008/). Use `flake8` or `pylint` for static checks.**
 * **Style requirements: The length of a single line of code cannot exceed 120 characters. If the length of a function exceeds 30 lines, split the code to improve readability.**
 * **Comment requirements: Complex logic and public interfaces must be commented. Modules, classes, and key functions must be described with docstrings, covering the usage, parameters, and return values.**
 * **Exception handling: Handle exceptions correctly. Do not swallow exceptions without handling or recording them. Resources must be released for critical paths.**

### 2.2 C++ Code Specifications

 * **Consistent Style: Comply with the existing coding style of the project and keep consistent with the surrounding code.**
 * **Naming Rules: Class names and structures use big camel case (for example, `DataManager`), and function names use small camel case (for example, `parseData`).**
 * **Comment requirements: Complex logic and public interfaces must be commented out to describe functions, parameters, and return values.**
 * **Exception handling: Handle exceptions correctly. Do not swallow exceptions without handling or recording them. RAII is used to obtain resources to ensure exception safety.**

## 3. Document Specifications

 * **Concise Expression: Use concise and clear expressions to avoid ambiguity and redundancy, and maintain unified technical terms.**
 * **Clear Structure: The title level is clear, the chapters are properly divided, and important conclusions or precautions can be highlighted.**
 * **Complete Example: Provide complete example code that can run, and specify the running environment or dependencies. Key steps are provided with descriptions.**
 * **Graphical and Textual: The complex processes, configuration items, or GUI operations must be provided with necessary diagrams for easy understanding.**

## 4. Pull Request Specifications

 * **Moderate Volume: A PR should not be too large. This facilitates the reviewer's understanding and quick feedback.**
 * **Single responsibility: One PR only solves one problem or implements one function, facilitating backtracking and combination.**
 * **Timely response: After receiving the review comments, reply or modify the review comments in a timely manner to avoid blocking the merge process.**

<br>

## 5. Pull Request Process Description

### 5.1 Checklist Before Submission

Before submitting a Pull Request, make sure that:

 * [ ] The code complies with the coding specifications of the project.
 * [ ] Add the necessary test cases.
 * [ ] All tests passed.
 * [ ] Update the related documentation.
 * [ ] The submission information is clear and concise.
 * [ ] The code has been self-reviewed.

### 5.2 Creating a PR

When creating a pull request on [GitCode](https://gitcode.com/), fill in the following information:

1. **Title: Briefly outline the theme or function of the change.**
2. **Description: Describe the changes, reasons, and self-test (including the environment and results) to facilitate the reviewer's understanding.**
3. **Associated Issues: PRs should be associated with issues to facilitate tracing.**

### 5.3 Code Review

1. After the PR is submitted, the reviewers and committers review the content.
2. Modify the code based on the review comments and push the update. Multiple iterations may be required. Please respond in a timely manner and keep communication.

### 5.4 Code Consolidation

The PR must obtain the following four labels in sequence before it can be incorporated into the mainline:

| Label                 | Description                                                                                                                                           |
| --------------------  | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| ascend-cla/yes        | **CLA signing: The CLA must be signed for the first contribution. This tag will be automatically obtained when the contribution is submitted later.** |
| ci-pipeline-passed    | **CI Pass: Comment `compile` in the PR to trigger the pipeline. If the check fails, modify the information as prompted and submit it again.**                      |
| lgtm                  | **Reviewer Approval: Two reviewers comment `/lgtm` in the PR to approve it.**                                                |
| approved              | **Committer Approval: Committers comment `/approved` in the PR to approve it.**                                             |

Once the four labels are assembled, the PR will be merged into the mainline.
