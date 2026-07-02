# Document Compiling Specifications

This document is used to standardize the document writing format, logic, and organization structure in the msit repository. Continuously updated, you can discuss at any time if you have good opinions.

# Structure of the msit document directory

 * Root Directory
    
     * Home page of the README.md msit document, which describes the inference service and the functions of the msit tool in the inference service.
     * start-page.md The start page of the document, which is equivalent to the cover page and is the first impression of the user.
     * The menu.md menu document is the framework of the entire msit document and is displayed in the sidebar and navigation bar on the GUI.
     * index.html: home page, which is used to display the page and does not need to be changed.
     * doc-guidelines.md Current Document, Document Writing Specifications
     * /install Installation Documents
     * The /assets page displays the required JS and CSS files, which do not need to be changed.
     * /pics: stores the images required by the MD file.
     * The other directories contain the description documents of different tools.
        
         * README.md tool home page
         * Specific scenario and function description document of the xxx.md
         * FAQ.md FAQ
         * /history Historical version
         * /pics: stores images required by the MD file.

# writing logic

1. For an independent tool, you need to add the README.md file under the tool directory as the homepage of the tool. The file contains the following information:
    
    1. Function introduction: The function is introduced through the scenario, which is easier to understand. Generally, the tool corresponds to a large scenario. For example, llm corresponds to acceleration library optimization. Benchmark corresponds to inference running and evaluation. A large scenario may also contain several sub-scenarios. We need to make the scene clear.
        
        1. Upstream and downstream of the scenario; What is the user workflow?
        2. What problems may be encountered in the user workflow;
        3. What functions do we have to solve the problem?
        4. If the tool contains only one function and the scenario is simple, you can directly describe the function in the current document. If the mode is complex and involves multiple functions and scenarios, you can describe the functions or scenarios in an independent MD document. and provides a quick jump link in the current document. (For example, a precision comparison scenario, including dump and comparison, can be described in a separate document, or an independent function can be described in a separate document.)
    2. Tool Development Plan
    3. Related links (API list, other related tools, installation guide, FAQ, historical version, etc.)
    4. FAQ. If there are many FAQs, you can create an FAQ.md document. And provide redirection links
    5. If the tool needs to retain historical version documents, you can add the history folder and archive historical documents in the history directory. Add the corresponding link on the tool homepage.
2. This section describes a specific scenario in the tool. Add an md file. (xxxx scenario usage description.md and xxx scenario.md) include the following:
    
    1. The main scenario and main process are introduced first. For details about the special process, see the following chapters.
    2. Procedure for using the tool
        
        1. Prerequisites and Precautions
        2. Describe the internal process of using the formal tool to facilitate troubleshooting and user understanding.
        3. How to view and analyze tool results?
        4. Exception Description
    3. Description of some branch scenarios, including special scenarios and how to handle abnormal scenarios.
    4. The scenario document does not provide the API parameter list or command line parameter list. You can go to the function description document.
3. This topic describes a specific function of the tool. Add an md file. (xxxx Function Usage Description.md, xxx Function.md). Includes the following:
    
    1. This section describes the functions. The scenarios can be described briefly, or links can be added to the specific scenario document.
    2. API parameter list
        
         * Describe the parameter application scenario so that users can understand the scenario in which the parameter is used.
    3. Provide some simple examples or instructions, which can be placed before or after the parameter list.
    4. Descriptions must be added to the output of the function, so that users can understand how to perform subsequent operations after using the tool.
4. The functions and scenarios can be cross-described and referenced. Multiple angle description tool.
    
    1. The scenario description should combine the functions without introducing all aspects of the functions. Only the scenario-related functions should be included. If a single function can handle the entire scenario, describe the scenario clearly in the function description document.
    2. The function description contains all the capabilities of the function. Different capabilities may be used in different scenarios. The description needs to be clear.

# Writing format

1. API or command parameters
    
    1. A column is added to describe the supported versions. (You can leave this parameter blank. New parameters need to be set later.)
    2. If there are many parameters, add a column for parameter grouping.
    3. The parameter sequence should be the same as that in the --help command.
2. Others to be added...
