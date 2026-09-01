# Documentation Writing Guidelines

This document defines the writing format, writing logic, and organizational structure of documents in the msit repository. It is continuously updated. If you have good suggestions, feel free to discuss at any time.

# msit Documentation Directory Structure

- Root directory
  - README.md: msit documentation homepage, introducing the inference business and the role of msit tools in the inference business
  - start-page.md: The start page of the documentation, serving as a cover page, giving users their first impression
  - menu.md: Menu document, serving as the skeleton of the entire msit documentation, reflected in the sidebar and navigation bar in the interface
  - index.html: Interface homepage, responsible for interface display. No changes needed
  - doc-guidelines.md: Current document, documentation writing guidelines
  - /install: Installation-related documents
  - /assets: js and css files required for interface display, no changes needed
  - /pics: Stores images needed by md documents
  - Other directories contain documentation for different tools
    - README.md: Tool homepage
    - xxx.md: Specific scenario or feature description documents
    - FAQ.md: FAQ
    - /history: Historical versions
    - /pics: Stores images needed by md documents

# Writing Logic

1. For an independent tool, add a README.md document in the tool directory as the tool homepage. The content is a general introduction to the tool. It includes the following:
   1. Feature introduction: Introduce features through scenarios, making them easier to understand. A tool generally corresponds to a larger scenario. For example, llm corresponds to acceleration library tuning; benchmark corresponds to inference execution and evaluation. Large scenarios may also contain several sub-scenarios. We need to explain the scenarios clearly.
      1. The upstream and downstream of the scenario; what the user workflow looks like.
      2. What problems users may encounter in their workflow.
      3. Which features we have that can solve the problems.
      4. If the tool only contains a single feature and the scenario is simple, the feature usage can be explained directly in the current document. If it is complex and involves multiple features and multiple scenarios, the feature or scenario can be introduced in a separate md document. And provide a quick navigation link in the current document (for example, a precision comparison scenario that includes dump and comparison can be described in a separate document. Or a relatively independent feature can be described in a separate document).
   2. Tool development plan
   3. Related links (API list, other related tools, installation guide, FAQ, historical versions, and so on)
   4. FAQ. If there are many FAQs, a separate FAQ.md document can be created. And provide a navigation link.
   5. If the tool needs to retain historical version materials, a history folder can be added, and historical documents can be archived in the history directory. And add the corresponding link on the tool homepage.
2. A specific scenario introduction for a tool. Add an md document (xxxx_scenario_usage_guide.md, xxx_scenario.md) including the following:
   1. Specific scenario description. Generally, introduce the main scenario and main process first. Special processes are introduced in later sections.
   2. Tool usage steps
      1. Prerequisites and precautions
      2. Formal tool usage steps. It is recommended to introduce the internal process of the tool to facilitate troubleshooting and user understanding.
      3. How to view and analyze tool results
      4. Exception description
   3. Description of some branch scenarios, including how to handle special scenarios and exception scenarios.
   4. In scenario documents, do not directly provide API parameter lists or command line parameter lists. You can navigate to the feature description document.
3. A specific feature introduction for a tool. Add an md document (xxxx_feature_usage_guide.md, xxx_feature.md). It includes the following:
   1. Mainly describe features. Scenarios can be briefly described or linked to specific scenario documents.
   2. API parameter list
      * The parameter usage scenario needs to be briefly described so that users can understand in which scenario the parameter is used when they see it.
   3. Provide some simple samples or usage instructions, which can be placed before or after the parameter list.
   4. The output of the feature needs to be explained so that users can understand what to do next after using the tool.
4. Features and scenarios can be cross-referenced and described from multiple angles.
   1. Scenario descriptions need to connect features. There is no need to introduce all aspects of features. Only include what is relevant to the scenario. If a single feature can handle the entire scenario, describe the scenario clearly in the feature description document.
   2. Feature descriptions include all capabilities of the feature. Different capabilities may be applied in different scenarios. They need to be described clearly.

# Writing Format

1. API or command parameters

   1. A column needs to be added to indicate the supported version (historical ones can be left blank, and newly added parameters need to be filled in).
   2. If there are many parameters, a column needs to be added for parameter grouping.
   3. The order of parameters should be consistent with that in --help.

2. Others to be supplemented...
