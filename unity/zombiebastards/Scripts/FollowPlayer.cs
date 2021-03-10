using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FollowPlayer : MonoBehaviour
{
    public GameObject player;
    private Vector3 offsetPos;
    // private Quaternion offsetRot = new Quaternion(32, -3, 0);
    // Start is called before the first frame update
    void Start()
    {
        offsetPos = new Vector3
        (transform.position.x - player.transform.position.x, 
        transform.position.y - player.transform.position.y, 
        transform.position.z - player.transform.position.z);
    }

    // Update is called once per frame
    void Update()
    {
        transform.position = player.transform.position + offsetPos;
        // transform.rotation = new Quaternion(0, 0, 0, 0);
    }
}
