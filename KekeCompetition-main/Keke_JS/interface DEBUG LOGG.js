// INTERFACE API FOR LLM-BASED BABA AGENTS

function logToFile(message) {
    const logFilePath = './interface_logs.txt'; // Log file in the current working directory
    const timestamp = new Date().toISOString();
    const logMessage = `[${timestamp}] ${message}\n`;
    fs.appendFileSync(logFilePath, logMessage, 'utf8');
}

function loadJsonFromFile(filePath) {
    try {
        // Read the file synchronously
        const rawData = fs.readFileSync(filePath, 'utf8');

        // Parse the JSON data
        const jsonData = JSON.parse(rawData);

        logToFile(`Loaded state from file: ${JSON.stringify(jsonData, null, 2)}`);
        return jsonData;
    } catch (error) {
        logToFile(`Error reading or parsing file: ${error}`);
        return null;
    }
}

// Modified from ./js/json_io.js
function saveJsonToFile(jsonData, filePath) {
    // Convert JSON object to string
    const data = JSON.stringify(jsonData, null, 2);

    logToFile(`Saving state to file: ${JSON.stringify(jsonData, null, 2)}`);
    // Write JSON string to a file
    fs.writeFile(filePath, data, (err) => {
      if (err) throw err;
    });
}

function importLevels(lvlsetJSON, levelsDir){
    let path = levelsDir + '/' + lvlsetJSON + ".json";
    if (!fs.existsSync(path)){return null;}     //no level set found

    let j = fs.readFileSync(path);
    let lvlset = JSON.parse(j);
    return lvlset.levels;
}

//file I/O
const fs = require('fs');

//import simulation code
const simjs = require('./js/simulation');

/**
 * Process an action and update the game state.
 * @param {string} action - The action to be performed (e.g., 'up', 'down', 'left', 'right', 'space').
 * @return {Object} The updated game state and any relevant information (e.g., whether the level is won).
 */
function processAction(action, gameState) {
    logToFile(`Processing action: ${action}`);
    logToFile(`GameState before action: ${JSON.stringify(gameState, null, 2)}`);

    let actionResult = simjs.nextMove(action, gameState);
    let newGameState = actionResult.next_state;

    logToFile(`GameState after action: ${JSON.stringify(newGameState, null, 2)}`);
    logToFile(`Won status: ${actionResult.won}`);

    // Return the updated state and whether the level is won
    return {
        state: newGameState,
        won: actionResult.won
    };
}

let args = process.argv;
let statePath = args[2];
let resultPath = args[3];
let levelSet = args[4]; // e.g., demo_LEVELS
let levelID = parseInt(args[5]);
let move = args[6];

let levelsDir = 'KekeCompetition-main/Keke_JS/json_levels'

let gameState;

if (statePath === 'None') {
    // Load levels
    const levels = importLevels(levelSet, levelsDir);
    // Convert ASCII map string to 2D array format
    let initialMap = simjs.parseMap(levels[levelID].ascii);
    logToFile(`Initial ASCII Map: ${levels[levelID].ascii}`);
    // Initialize game state
    gameState = simjs.newState(initialMap);
    logToFile(`Initialized GameState: ${JSON.stringify(gameState, null, 2)}`);
} else {
    gameState = loadJsonFromFile(statePath).state;
    logToFile(`Loaded GameState: ${JSON.stringify(gameState, null, 2)}`);
}

if (move !== 'None') {
    let result = processAction(move, gameState);
    saveJsonToFile(result, resultPath);
    logToFile(`Final GameState after move: ${JSON.stringify(result.state, null, 2)}`);
    logToFile(`Game won: ${result.won}`);
} else {
    saveJsonToFile(gameState, resultPath);
    logToFile(`Saved GameState without move: ${JSON.stringify(gameState, null, 2)}`);
}
