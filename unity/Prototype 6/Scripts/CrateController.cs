using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CrateController : MonoBehaviour
{
    public float moveSpeed = 5f;
    public float borderZ = 15; // Left/right 'ends' of scene
    public float borderOffset = 5; // Amount the crate is moved when borderZ is hit
    private Rigidbody playerRb;
    // Start is called before the first frame update
    void Start()
    {
        playerRb = GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetAxis("Mouse X") > 0)
        {
            playerRb.AddForce(Vector3.forward * moveSpeed * Time.deltaTime, ForceMode.VelocityChange);
        }
        else if (Input.GetAxis("Mouse X") < 0)
        {
            playerRb.AddForce(Vector3.back * moveSpeed * Time.deltaTime, ForceMode.VelocityChange);
        }

        if (transform.position.z < -borderZ)
        {
            transform.position = new Vector3(transform.position.x, transform.position.y, transform.position.z + borderOffset);
        }

        if (transform.position.z > borderZ)
        {
            transform.position = new Vector3(transform.position.x, transform.position.y, transform.position.z - borderOffset);
        }
    }
}
