using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SpawnObstacle : MonoBehaviour
{
    public GameObject[] obstacles;
    public GameObject player;
    public PlayerController playerController;
    private bool gameOverCall;
    private float lowerXZBound = -10f;
    private float upperXZBound = 10f;
    // Start is called before the first frame update
    void Start()
    {
        playerController = player.GetComponent<PlayerController>();
        InvokeRepeating("SpawnObstacles", 0, 5);
    }

    // Update is called once per frame
    void Update()
    {
        gameOverCall = playerController.gameOver;

        if (gameOverCall)
        {
            CancelInvoke();
        }
    }

    void SpawnObstacles()
    {
        Vector3 spawnPos = new Vector3(Random.Range(lowerXZBound, upperXZBound), 0.8f, Random.Range(lowerXZBound, upperXZBound));
        int obstacleIndex = Random.Range(0, obstacles.Length);
        Instantiate(obstacles[obstacleIndex], spawnPos, obstacles[obstacleIndex].transform.rotation);
    }
}

// This script instantiates an obstacle (either in horizontal or vertical rotation) every x seconds in order to make it harder for the player
// to navigate the scene over time. Once the game is over, the spawning iteration stops.
