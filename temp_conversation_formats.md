================================================================================
# Simultaneous Action and Conversation:

# Game Description:
This game is all about chatting and having fun!

# History:
## Round 0 Observation:

Messages From Players:
No messages.

Previous Round Reward: [None, None]

Possible Actions:
[0]: Cook a delicious pie.
[1]: Play a game of poker with the players around you.
[2]: Do some exercise to stay fit.
[3]: Do nothing.

## Round 0 My Action: None

## Round 1 Observation:

Messages From Players:
Alice: "Hey everyone! Who's up for a game of poker? Let's have some fun together!"
(Me) Bob: "Who's up for a thrilling game of poker?"

Previous Round Reward: [0, 0]

Possible Actions:
[0]: Cook a delicious pie.
[1]: Play a game of poker with the players around you.
[2]: Do some exercise to stay fit.
[3]: Do nothing.

## Round 1 My Action: Play a game of poker with the players around you.

## Round 2 Observation:

Messages From Players:
Alice: "Hey, who's ready for another round of poker? Let's keep the fun going!"
(Me) Bob: "Let's keep the excitement going with another round of poker!"

Previous Round Reward: [0, 0]

Possible Actions:
[0]: Cook a delicious pie.
[1]: Play a game of poker with the players around you.
[2]: Do some exercise to stay fit.
[3]: Do nothing.

## Round 2 My Action: Play a game of poker with the players around you.

# Expectations:
Given the current dynamics of the game and the interactions between players so far, I have updated expectations for each player in this round.

1. **Alice**: Alice, being enthusiastic about playing poker and fostering social interactions, is likely to propose another round of poker to continue the fun and engagement with the other players. She may actively participate in the poker game again to enjoy the social aspect of the activity.

2. **Bob (Me)**: As I have been actively engaged in playing poker and initiating game rounds in the past, I am expected to continue participating in the poker game to maintain the lively atmosphere and have an enjoyable time with the other players.

3. **Cathy**: Given that no one has cooked a delicious pie yet in the game and the option remains available, Cathy might decide to showcase her culinary skills by choosing to cook a delicious pie in this round. This action could introduce a new dynamic to the game and provide a delightful treat for all players.

4. **David**: David, who values physical well-being and fitness, is likely to opt for doing some exercise in this round to maintain his health and energy levels. Engaging in physical activities could be a priority for David, aligning with his focus on staying fit and active.

5. **Eve**: Eve, who enjoys social interactions and fun experiences, may choose to join the game of poker once again to bond with others and have an entertaining time. Participating in the poker game could offer Eve a chance to engage with fellow players and enjoy the interactive aspect of the activity.

6. **Neutral Players**: Some players may continue to choose the option of doing nothing, indicating a preference for a relaxed approach to the game or perhaps observing the ongoing interactions before deciding to engage actively. Their actions could vary based on individual preferences and the evolving dynamics of the game.

In this round, a mix of actions such as playing poker, cooking a delicious pie, doing exercise, and doing nothing is anticipated based on the players' personalities, interests, and previous interactions. These diverse choices are expected to contribute to a dynamic and engaging gameplay experience for all participants.

# Current Round:
You may both select an action and say something to the players around you.
What is your thought process on selecting this round's action? Once you have fully finished explaining your thought process, first output the message you would like to send to nearby players and then output the index of your final action selection surrounded by quotation marks.
If you don't wish to say anything to other players, simply output an empty string.

Example output format:
'This is my thought process...
Message To Other Player: "<my_message>"
Final Answer: "<action_selection>".'



================================================================================
# Abandonned Chat Formats:

Round 1.1: Alice: "Hey everyone! Who's up for a game of poker? Let's have some fun together!"
Round 1.1: (Me) Bob: "Who's up for a thrilling game of poker?"
Round 1.2: Alice: "Hey everyone! Who's up for a game of poker? Let's have some fun together!"
Round 1.2: (Me) Bob: "Who's up for a thrilling game of poker?"

================================================================================
# Chatting and Actions Phases

1. Round 1 Chat (n chat rounds)
2. Round 1 Action Selection
3. Round 2 Chat (n chat rounds)
4. Round 2 Action Selection
5. ...

- Within each discussion phase, agents can either all send their messages at the same time, OR they can submit their message sequentially 

================================================================================
# Current Best Format:
## No History Formats:
### History:
No history, this is the first round.

### Chat:
#### No History At All:
----- Discussion Phase 1/5 -----
No chat History

#### No Round History:
----- Discussion Phase 1/5 -----
Alice: "Hey everyone! Who's up for a game of poker? Let's have some fun together!"
(Me) Bob: "Who's up for a thrilling game of poker?"

----- Discussion Phase 2/5 -----
No chat History

================================================================================
## Asking For Conversation:

This is a turn based game where players talk then choose what actions they want to do.

# Game Description:
This game is all about chatting and having fun!

# History:
No history, this is the first round.

# Chat:
----- Discussion Phase 1/5 -----
Alice: "Hey everyone! Who's up for a game of poker? Let's have some fun together!"
(Me) Bob: "Who's up for a thrilling game of poker?"

----- Discussion Phase 2/5 -----
Alice: "Hey everyone! Who's up for a game of poker? Let's have some fun together!"

# Current Discussion Phase 
What do you want to say (You have a 120 character limit)?

================================================================================
## Asking For Action:

This is a turn based game where players talk then choose what actions they want to do.

# Game Description:
This game is all about chatting and having fun!

# History:
No history, this is the first round.

# Chat:
----- Discussion Phase 1/5 -----
Alice: "Hey everyone! Who's up for a game of poker? Let's have some fun together!"
(Me) Bob: "Who's up for a thrilling game of poker?"

----- Discussion Phase 2/5 -----
Alice: "Hey everyone! Who's up for a game of poker? Let's have some fun together!"
(Me) Bob: "Who's up for a thrilling game of poker?"

----- Discussion Phase 3/5 -----
Alice: "Hey everyone! Who's up for a game of poker? Let's have some fun together!"
(Me) Bob: "Who's up for a thrilling game of poker?"

----- Discussion Phase 4/5 -----
Alice: "Hey everyone! Who's up for a game of poker? Let's have some fun together!"
(Me) Bob: "Who's up for a thrilling game of poker?"

----- Discussion Phase 5/5 -----
Alice: "Hey everyone! Who's up for a game of poker? Let's have some fun together!"
(Me) Bob: "Who's up for a thrilling game of poker?"

# Current Action Phase
Possible Actions:
[0]: Cook a delicious pie.
[1]: Play a game of poker with the players around you.
[2]: Do some exercise to stay fit.
[3]: Do nothing.

What is your thought process on selecting this round's action? Once you have fully finished explaining your thought process, output the index of your final action selection surrounded by quotation marks (e.g. Final Answer: "<action_selection>").

================================================================================
# Examples With Multiple Rounds of History:

================================================================================
## Decomposed Observations (Asking For Action):

This is a turn based game where players talk then choose what actions they want to do.

# Game Description:
This game is all about chatting and having fun!

# Round 0
## Round 0 Environment Observation:
You are at the farm. You see...

## Round 0 Discussion Phase:
----- Discussion Phase 1/5 -----
Alice: "Hey everyone! Who's up for a game of poker? Let's have some fun together!"
(Me) Bob: "Who's up for a thrilling game of poker?"

----- Discussion Phase 2/5 -----
Alice: "Hey everyone! Who's up for a game of poker? Let's have some fun together!"
(Me) Bob: "Who's up for a thrilling game of poker?"

----- Discussion Phase 3/5 -----
Alice: "Hey everyone! Who's up for a game of poker? Let's have some fun together!"
(Me) Bob: "Who's up for a thrilling game of poker?"

----- Discussion Phase 4/5 -----
Alice: "Hey everyone! Who's up for a game of poker? Let's have some fun together!"
(Me) Bob: "Who's up for a thrilling game of poker?"

----- Discussion Phase 5/5 -----
Alice: "Hey everyone! Who's up for a game of poker? Let's have some fun together!"
(Me) Bob: "Who's up for a thrilling game of poker?"

## Round 0 Actions:
Alice: Play poker
(Me) Bob: Eat pie

## Round 0 Reward:
My reward: 5 points

# (Current Round) Round 1
## (Current Round) Round 1 Environment Observation:
You are at the farm. You see...

## (Current Round) Round 1 Discussion Phase:
----- Discussion Phase 1/5 -----
Alice: "Hey everyone! Who's up for a game of poker? Let's have some fun together!"
(Me) Bob: "Who's up for a thrilling game of poker?"

----- Discussion Phase 2/5 -----
Alice: "Hey everyone! Who's up for a game of poker? Let's have some fun together!"
(Me) Bob: "Who's up for a thrilling game of poker?"

----- Discussion Phase 3/5 -----
Alice: "Hey everyone! Who's up for a game of poker? Let's have some fun together!"
(Me) Bob: "Who's up for a thrilling game of poker?"

----- Discussion Phase 4/5 -----
Alice: "Hey everyone! Who's up for a game of poker? Let's have some fun together!"
(Me) Bob: "Who's up for a thrilling game of poker?"

----- Discussion Phase 5/5 -----
Alice: "Hey everyone! Who's up for a game of poker? Let's have some fun together!"
(Me) Bob: "Who's up for a thrilling game of poker?"

## (Current Round) Round 1 Action Phase:
Possible Actions:
[0]: Cook a delicious pie.
[1]: Play a game of poker with the players around you.
[2]: Do some exercise to stay fit.
[3]: Do nothing.

What is your thought process on selecting this round's action? Once you have fully finished explaining your thought process, output the index of your final action selection surrounded by quotation marks (e.g. Final Answer: "<action_selection>").

================================================================================
## Everything Is In Observation (Asking For Conversation):

- we can just have the observation include the round number and be decomposed as shown above.
	- Ie., the observation would show the subtitles:
		- Round n Environment Observation
		- Round n Discussion Phase
		- Round n-1 Player Actions
		- Round n-1 Rewards
		- Possible Actions
	- the only diff would be that we would be missing the "(Current Round)" text for the round
	- nearly all agents would need to ignore this format and simply use the raw info provided by the Observation
		- I think this is ok since the Observation MUST show info which is only really relevant for the current round (ex., the possible actions or things like your current inventory state). The Observation shows the max amount of information, and it is up to the Agent to figure out what/if it wants to filter out info



This is a turn based game where players talk then choose what actions they want to do.

# Game Description:
This game is all about chatting and having fun!

# Round 0
## Round 0 Environment Observation:
You are at the club house. You see Alice.

## Round 0 Discussion Phase:
----- Discussion Phase 1/3 -----
Alice: "Hey Bob, hows it going?"
(Me) Bob: "I'm doing well, how about you?"

----- Discussion Phase 2/3 -----
Alice: "Ya, not bad. I think I'm going to play poker, want to join me?"
(Me) Bob: "Sure, poker sounds fun!"

----- Discussion Phase 3/3 -----
Alice: "Great! I'll setup the cards."
(Me) Bob: "Thanks Alice, I'll set some chips for us to play with."

## Round 0 Actions:
Alice: Cook a delicious pie.
(Me) Bob: Play a game of poker.

## Round 0 Reward:
My reward: -5 points

# (Current Round) Round 1
## Round 1 Environment Observation:
You are at the club house. You see Alice.

## Round 1 Discussion Phase:
----- Discussion Phase 1/3 -----
Alice: "Hey, Bob want some pie?"
(Me) Bob: "Alice no! Why did you bake a pie when you said you would play poker?"

----- Discussion Phase 2/3 -----
Alice: "I got bored."

# Current Discussion Phase 
What do you want to say (You have a 120 character limit)?

================================================================================
## Implementation

- each step we must loop over all agents an extra dicussion_turns_num of times, so that each agent gets to say something for each turn
- should we use an env.submit_message() method?
	- alternatives? No matter what we do, we need some kind of mini-step function, so I think it doesn't matter too much
- env.submit_message() will simply add to a list/history of messages, the person running the env is responsible for deciding how they want to submit messags.
	- the person running the env controls for agent speaking order by deciding the order in which env.submit_message() is called	
		- this can be packageed in a run function which presets some orders (ex., allways the same order, diff order each time, etc.) if we want
	- the person running the env can also control agents skipping their right to call env.submit_message()


================================================================================
# OLD: Simulatanious Chat And Action

# History:
## Round 0
## Round 0 Observation:

Chat:
No messages.

Previous Round Reward: [None, None]

Possible Actions:
[0]: Cook a delicious pie.
[1]: Play a game of poker with the players around you.
[2]: Do some exercise to stay fit.
[3]: Do nothing.

## Round 0 My Action: None

## Round 1 Observation:

Messages From Players:
Alice: "Hey everyone! Who's up for a game of poker? Let's have some fun together!"
(Me) Bob: "Who's up for a thrilling game of poker?"

Previous Round Reward: [0, 0]

Possible Actions:
[0]: Cook a delicious pie.
[1]: Play a game of poker with the players around you.
[2]: Do some exercise to stay fit.
[3]: Do nothing.

## Round 1 My Action: Play a game of poker with the players around you.

## Round 2 Observation:

Messages From Players:
Alice: "Hey, who's ready for another round of poker? Let's keep the fun going!"
(Me) Bob: "Let's keep the excitement going with another round of poker!"

Previous Round Reward: [0, 0]

Possible Actions:
[0]: Cook a delicious pie.
[1]: Play a game of poker with the players around you.
[2]: Do some exercise to stay fit.
[3]: Do nothing.

## Round 2 My Action: Play a game of poker with the players around you.

# Expectations:
Given the current dynamics of the game and the interactions between players so far, I have updated expectations for each player in this round.

1. **Alice**: Alice, being enthusiastic about playing poker and fostering social interactions, is likely to propose another round of poker to continue the fun and engagement with the other players. She may actively participate in the poker game again to enjoy the social aspect of the activity.

2. **Bob (Me)**: As I have been actively engaged in playing poker and initiating game rounds in the past, I am expected to continue participating in the poker game to maintain the lively atmosphere and have an enjoyable time with the other players.

3. **Cathy**: Given that no one has cooked a delicious pie yet in the game and the option remains available, Cathy might decide to showcase her culinary skills by choosing to cook a delicious pie in this round. This action could introduce a new dynamic to the game and provide a delightful treat for all players.

4. **David**: David, who values physical well-being and fitness, is likely to opt for doing some exercise in this round to maintain his health and energy levels. Engaging in physical activities could be a priority for David, aligning with his focus on staying fit and active.

5. **Eve**: Eve, who enjoys social interactions and fun experiences, may choose to join the game of poker once again to bond with others and have an entertaining time. Participating in the poker game could offer Eve a chance to engage with fellow players and enjoy the interactive aspect of the activity.

6. **Neutral Players**: Some players may continue to choose the option of doing nothing, indicating a preference for a relaxed approach to the game or perhaps observing the ongoing interactions before deciding to engage actively. Their actions could vary based on individual preferences and the evolving dynamics of the game.

In this round, a mix of actions such as playing poker, cooking a delicious pie, doing exercise, and doing nothing is anticipated based on the players' personalities, interests, and previous interactions. These diverse choices are expected to contribute to a dynamic and engaging gameplay experience for all participants.

# Current Round:
You may both select an action and say something to the players around you.
What is your thought process on selecting this round's action? Once you have fully finished explaining your thought process, first output the message you would like to send to nearby players and then output the index of your final action selection surrounded by quotation marks.
If you don't wish to say anything to other players, simply output an empty string.

Example output format:
'This is my thought process...
Message To Other Player: "<my_message>"
Final Answer: "<action_selection>".'