# Contribution Process and Specifications

Thank you for considering contributing! We welcome contributions in any form, including bug fixes, feature enhancements, documentation improvements, and even simple usage feedback.
Whether you're an experienced developer or a newcomer to an open source project for the first time, your contributions will be invaluable.

You can support this project in the following ways:

 * **Code contribution: Fix known bugs, optimize performance, refactor improvements, or implement new functions.**[Contribution process](#1-contribution-process)    And to the[Code Specifications](#2-code-specifications)    Submit the code.
 * **Issue feedback: Report bugs through issue, raise function suggestions or questions, or participate in requirement review and solution discussion.**
 * **Document improvement: Correct document errors, supplement missing content, optimize description, or write examples and tutorials. Please follow the**[Document Specifications](#3-document-specifications)    .
 * **Quality assurance: Supplement or optimize test cases, review Pull Requests, provide constructive suggestions, and assist other contributors in improving code quality.**
 * **Community promotion: Answer questions, share usage experience, and best practices in the issue/PR, or write blogs, tutorials, and promote projects through publicize in technical communities and social media.**

## 1. Contribution process

1. **Derived repository: Fork source code repository to personal repository, and then clone personal repository to the local development environment.**
2. **Creating a branch: Create a functional branch based on the latest main branch. The branch name should be concise and reflect the changes (e.g.**`fix_xxx_bug`,`feature_xxx`).
3. **Code development: Perform development on the function branch.**[Code Specifications](#2-code-specifications)    and keep submission records clear and atomic.
4. **Local test: Verify the functions of the code and supplement unit tests as required based on the development module situation to ensure that all tests are passed.**
5. **Document Update: Supplement or update the documents related to the change.**[Document Specifications](#3-document-specifications)    .
6. **Request to merge: Submit a PR. Please comply with the requirements.**[Pull Request Specification](#4-pull-request-specifications)    For details, see.[Description of the Pull Request process](#5-pull-request-process-description)    .
7. **Tracking and incorporation: Track the Pull Request progress, respond to review comments in a timely manner, and modify the code until the code passes the review and is incorporated into the mainline.**

## 2. Code Specifications

### 2.1 Python Code Specifications

 * **Coding specifications: Comply with**[PEP 8](https://peps.python.org/pep-0008/), recommended`flake8`or the`pylint`Perform static checks.
 * **Style requirements: The length of a single line of code cannot exceed 120 characters. If the length of a function exceeds 30 lines, split the code to improve readability.**
 * **Comment requirements: Complex logic and public interfaces must be commented out. Modules, classes, and key functions must be described with docstring to describe the usage, parameters, and return values.**
 * **Exception handling: Handle exceptions correctly. Do not swallow exceptions without handling or recording them. Resources must be released for critical paths.**

### 2.2 C++ Code Specifications

 * **Consistent style: Comply with the existing coding style of the project and keep consistent with the surrounding code.**
 * **Naming rules: class name and structure big camel case (e.g.**`DataManager`), function name small camel case (e.g.`parseData`).
 * **Comment requirements: Complex logic and public interfaces must be commented out to describe functions, parameters, and return values.**
 * **Exception handling: Handle exceptions correctly. Do not swallow exceptions without handling or recording them. RAII is used to obtain resources to ensure abnormal security.**

## 3. Document Specifications

 * **Concise expression: Use concise and clear Chinese expressions to avoid ambiguity and redundancy, and maintain unified technical terms.**
 * **Clear structure: The title level is clear, the chapters are properly divided, and important conclusions or precautions can be highlighted.**
 * **Complete example: provides complete example code that can be runnable and specifies the running environment or dependency. Key steps are provided with descriptions.**
 * **Graphical and textual: The complex processes, configuration items, or GUI operations must be provided with necessary diagrams for easy understanding.**

## 4. Pull Request Specifications

 * **Moderate volume: A PR change should not be too large to facilitate the reviewer's understanding and quick feedback.**
 * **Single responsibility: One PR only solves one problem or implements one function, facilitating backtracking and combination.**
 * **Timely response: After receiving the review comments, reply or modify the review comments in a timely manner to avoid blocking the merge process.**

## 5. Pull Request Process Description

### 5.1 Checklist Before Submission

Before submitting a Pull Request, make sure that:

 * \[\] The code complies with the coding specifications of the project.
 * \[\] added the necessary test cases.
 * \[\] All tests passed.
 * \[\] updated related documentation.
 * \[\] The submission information is clear and clear.
 * The \[\] code has been self-reviewed.

### 5.2 Create PR

In the[GitCode](https://gitcode.com/)    When creating a pull request on the, fill in the following information:

1. **Title: Briefly outline the theme or function of the change.**
2. **Description: Describe the changes, reasons, and self-test (including the environment and results) to facilitate the reviewer's understanding.**
3. **Associated issues: PRs should be associated with issues to facilitate tracing.**

### 5.3 Code review

1. After the PR is submitted, the reviewers and committers review the content.
2. Modify the code based on the review comments and push the update. Multiple iterations may be required. Please respond in a timely manner and keep communication.

### 5.4 Code Consolidation

The PR must obtain the following four labels in sequence before it can be incorporated into the trunk:

| Label                | Description                                                                                                                                           |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ascend-cla/yes`     | **CLA signing: The CLA must be signed for the first contribution. This tag will be automatically obtained when the contribution is submitted later.** |
| `ci-pipeline-passed` | **CI Pass: Comment in PR**`compile`Trigger pipeline; If the check fails, modify the information as prompted and submit it again.                      |
| `lgtm`               | **Reviewer approval: Two reviewers submit comments in the PR after the reviewer approval.**`/lgtm`Get.                                                |
| `approved`           | **Committer approved: Committers submit comments in the PR after the review is approved.**`/approved`Get.                                             |

Once the four labels are assembled, the PR will be merged into the backbone branch.
