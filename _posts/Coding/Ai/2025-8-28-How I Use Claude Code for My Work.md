![title image](https://images.pexels.com/photos/414612/pexels-photo-414612.jpeg)

# Background

I am currently a full-time algorithm engineer, coding with Python and C++ daily. Recently, the leadership of my company mandated all developers use Claude Code for programming. The logic behind this is to use AI tools as leverage to boost the productivity of algorithm development, so that small teams can compete with large Internet companies. I've tried on the Ai tool and try to accumulate some experience and summarize workflows.

P.S. Chats between CC and me are surrounded by "quote" mark.

# Workflow

The whole process consists of 6 stages: Setups, Requirement Discussion, Plan making, Implementation, Code review, Dryrun/Deployment.

## 0. Setups/Prerequisites

To use Claude Code(CC), I use IDE VS-Code and extension 'Claude Code for VS-Code'. Open the project directory, use `/init` command to generate a CLAUDE.md for current repo. This helps prepare CC for this project. Get your MCPs and tools ready at this stage also. I also changed output-style to "Explanatory".

## 1. Requirement Discussion

First write a draft requirement document of new feature. Always start in "Plan mode".

> Read the current requirements document @file_path, and then I will ask questions based on the requirements.

When I first run into a problem, some parts of the plan I have not been thoroughly thought through.
So ask CC for recommendation, based on its understanding of the code repository.

> Where should I put the Protos, such that compatible with current coding style?

> When should the Redis read/write functions be triggered?

Give feedbacks on CC's responses, type `Esc` to stop generation and ask it to do something else.

## 2. Plan Making

> Okay, is there anything you don't understand about the current requirements document? Communicate with me to clarify it according to importance.

CC respond with many detailed questions (accompanied by importance level), I will answer them one by one.

After it is clear with all the developing plan, ask it to write down the details into the plan markdown @plan.md and resolve conflicts if any.

## 3. Implementation

`/clear` to clean up the history and free up context. Then ask CC to really develop based on current development plan.

## 4. Code Review

Continuously compile, report errors, pass error information to CC, and then fix, repeating the cycle until success.

## 5. Dryrun/Deployment

Go through the actual dryrun test process, and go online with the development shuttle after everything goes smoothly.

# Thoughts

- Structured intermediate outputs help, and iterative validate of the intermediate outputs yourself.
- Different agents should be assigned different contexts, views, memories, and tools, which can fulfill different roles. Some require understanding the overall architecture, while others require specific implementation details.
- The difference between agents and traditional implementation: Agents can integrate with the human review process, similar to a highway with toll booths, while the traditional programming is a winding, complex, narrow road with many forks. Highways allow us to go further, but they require higher road infrastructure, such as wide and smooth roads (corresponding to the construction of MCP and tools).
- When designing memory (as an important component of LLM agent), in addition to layering, a recursive structure is also necessary. That is, after considering a specific point locally, return the results back to the previous level, which then reviews and considers all the information. Then, I decide whether it's necessary to reconsider a specific point. The key here is to avoid falling into an endless loop.
- AI is powerful, but never too powerful. Just as, despite Moore's Law, software complexity has historically increased, leading to insufficient performance for hardware of mobile phones and computers, the same holds true for AI. As AI expands each individual's personal boundaries, the effort required of us to be competitive remains the same. Our outputs must be differentiated from the mass-produced output of ordinary LLMs to be valuable.
