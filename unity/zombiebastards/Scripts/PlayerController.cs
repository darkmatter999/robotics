using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerController : MonoBehaviour
{
    public GameObject leftLeg;
    public GameObject rightLeg;
    private float legRotateSpeed = 40f;
    private float soldierRotateSpeed = 100f;
    private float speed = 20.0f;
    private float horizontalInput;
    private float verticalInput;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        horizontalInput = Input.GetAxis("Horizontal");
        verticalInput = Input.GetAxis("Vertical");
        transform.Translate(Vector3.forward * speed * Time.deltaTime * verticalInput);
        //transform.Translate(Vector3.right * speed * Time.deltaTime * horizontalInput);

        transform.Rotate(Vector3.up, soldierRotateSpeed * Time.deltaTime * horizontalInput);
        
        if (leftLeg.transform.rotation.x <=0f && leftLeg.transform.rotation.x > -0.3f)
        {
            leftLeg.transform.Rotate(Vector3.left, legRotateSpeed * Time.deltaTime * verticalInput);
            rightLeg.transform.eulerAngles = new Vector3(0, 0, 0);
            
        } else
        {   
            if (rightLeg.transform.rotation.x > -0.3f)
            {
            leftLeg.transform.eulerAngles = new Vector3(0.0001f, 0, 0);
            rightLeg.transform.Rotate(Vector3.left, legRotateSpeed * Time.deltaTime * verticalInput);
            } else 
            {
                leftLeg.transform.eulerAngles = new Vector3(0, 0, 0);
            }
        }
        
        //leftLeg.transform.Rotate(Vector3.left * legRotateSpeed * Time.deltaTime * verticalInput);
        //rightLeg.transform.Rotate(Vector3.right * legRotateSpeed * Time.deltaTime * verticalInput);
        
        
    }
}
