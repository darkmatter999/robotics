using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnemyMove : MonoBehaviour
{
    // Initialize an array of the four 'grounded' movements
    private Vector3[] movements = {Vector3.back, Vector3.forward, Vector3.right, Vector3.left};
    private int movementIndex;
    // Initialize the enemies' moving speed
    private float speed = 5f;
    // The enemies get a kind of sensor which prevents them going further than the boundaries of the actual plane.
    // Therefore we need to implement bounds on the X and Z axes.
    private float lowerXZBound = -10;
    private float upperXZBound = 10;
    // Initialize a timer that controls how long a particular randomized movement is executed by the enemy
    private float moveTimer;
    public bool moveActive = true;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        // Below 'Move' method generates a new random movement (out of the 'movements' array) every x seconds
        // In other words, the generated movement is 'valid' for x seconds
        //if (moveTimer < 0)
        //{
        //    Move();
        //}

        // Timing the enemy movements through a Coroutine --> leave the 'movement index' valid for a period in seconds defined in the
        // Coroutine method below
        if (moveActive)
        {
            movementIndex = Random.Range(0, movements.Length);
            moveActive = false;
            StartCoroutine(enemyChangeDirectionRoutine());
        }

        transform.Translate(movements[movementIndex] * speed * Time.deltaTime);
        
        // Decrease the moveTimer
        // moveTimer -= Time.deltaTime;
        // Once the enemy has reached either lower or upper bound of the plane, it becomes reinstated in a random position **on** the plane
        if (transform.position.x < lowerXZBound || transform.position.x > upperXZBound || transform.position.z < lowerXZBound || transform.position.z > upperXZBound)
        {
            transform.position = new Vector3(Random.Range(lowerXZBound+1, upperXZBound-1), 0, Random.Range(lowerXZBound+1, upperXZBound-1));
        }
    }

    IEnumerator enemyChangeDirectionRoutine()
    {
        yield return new WaitForSeconds(2);
        moveActive = true;
    }
}
