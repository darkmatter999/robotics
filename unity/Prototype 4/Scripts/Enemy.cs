using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Enemy : MonoBehaviour
{
    public float speed = 10.0f;
    private Rigidbody enemyRb;
    private GameObject player;
    // Start is called before the first frame update
    void Start()
    {
        enemyRb = GetComponent<Rigidbody>();
        player = GameObject.Find("Player");
    }

    // Update is called once per frame
    void Update()
    {
        // We need to subtract the enemy's position from the player's position to generate a force that makes the enemy follow (chase) the player
        // So as to reduce the 'aggressiveness' of the enemy's chase force, we normalize the resulting vector. This results in a balanced force
        // strength at all times, regardless of the actual distance between the player and the enemy.
        Vector3 lookDirection = (player.transform.position - transform.position).normalized;
        enemyRb.AddForce(lookDirection * speed);
        // Destroy enemy if it is about to fall off the cliff
        if (transform.position.y < -10)
        {
            Destroy(gameObject);
        }
    }
}
