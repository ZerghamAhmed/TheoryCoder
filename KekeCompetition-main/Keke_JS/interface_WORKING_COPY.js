// INTERFACE API FOR LLM-BASED BABA AGENTS

// File I/O and importing necessary modules
const fs = require('fs');
const simjs = require('./js/simulation');

/**
 * Load JSON data from a file.
 * @param {string} filePath - The path to the JSON file.
 * @return {Object} - The parsed JSON data.
 */
function loadJsonFromFile(filePath) {
    try {
        const rawData = fs.readFileSync(filePath, 'utf8');
        return JSON.parse(rawData);
    } catch (error) {
        console.error('Error reading or parsing file:', error);
        return null;
    }
}

/**
 * Save JSON data to a file.
 * @param {Object} jsonData - The JSON data to save.
 * @param {string} filePath - The path to the file where JSON data should be saved.
 */
function saveJsonToFile(jsonData, filePath) {
    const data = JSON.stringify(jsonData, null, 2);
    fs.writeFile(filePath, data, (err) => {
        if (err) throw err;
    });
}

/**
 * Import levels from a JSON file.
 * @param {string} lvlsetJSON - The name of the level set JSON file (without extension).
 * @param {string} levelsDir - The directory where level sets are stored.
 * @return {Array} - The levels data.
 */
function importLevels(lvlsetJSON, levelsDir){
    let path = levelsDir + '/' + lvlsetJSON + ".json";
    if (!fs.existsSync(path)) return null;

    let j = fs.readFileSync(path);
    return JSON.parse(j).levels;
}

/**
 * Process an action and update the game state.
 * @param {string} action - The action to be performed (e.g., 'up', 'down', 'left', 'right', 'space').
 * @param {Object} gameState - The current game state.
 * @return {Object} - The updated game state and whether the level is won.
 */
function processAction(action, gameState) {
    let actionResult = simjs.nextMove(action, gameState);
    let newGameState = actionResult.next_state;

    // Convert obj_map and back_map to ASCII representation
    let asciiMap = simjs.doubleMap2Str(newGameState.obj_map, newGameState.back_map);

    // Return the minimal state object
    return {
        ascii_map: asciiMap,
        won: actionResult.won
    };
}

// Main Execution
let args = process.argv;
let statePath = args[2];
let resultPath = args[3];
let levelSet = args[4]; // e.g., demo_LEVELS
let levelID = parseInt(args[5]);
let move = args[6];

let levelsDir = 'KekeCompetition-main/Keke_JS/json_levels';

let gameState;

if (statePath === 'None') {
    // Load levels
    const levels = importLevels(levelSet, levelsDir);
    if (!levels) {
        console.error('Level set not found!');
        process.exit(1);
    }
    
    // Convert ASCII map string to 2D array format and initialize the game state
    let initialMap = simjs.parseMap(levels[levelID].ascii);
    gameState = simjs.newState(initialMap);
} else {
    // Load the game state from the provided path
    gameState = loadJsonFromFile(statePath).state;
}

if (move !== 'None') {
    // Process the action and update the game state
    let result = processAction(move, gameState);

    // Save the minimal state to a file
    saveJsonToFile(result, resultPath);
} else {
    // Convert obj_map and back_map to ASCII representation for the initial state
    let asciiMap = simjs.doubleMap2Str(gameState.obj_map, gameState.back_map);

    // Save the minimal initial state
    let initialState = {
        ascii_map: asciiMap,
        won: false
    };

    saveJsonToFile(initialState, resultPath);
}
